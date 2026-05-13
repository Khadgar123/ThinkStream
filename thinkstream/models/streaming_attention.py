import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    BlockMask,
)


def generate_video_sliding_window_mask_mod(
    video_mask,
    attention_mask,
    window_size_n,
    recall_video_mask=None,
    recall_kv_mask=None,
):
    """
    Generates a mask_mod for flex_attention that handles:
    1. Padding (via attention_mask)
    2. Causal masking
    3. Ordinary video sliding window (retain current block + prev n-1 blocks)
    4. Recall evidence as a one-turn sidecar: visible inside the current
       ordinary visual timestep, but it does not consume ordinary window slots.
    5. Text tokens (always visible if valid unless explicitly marked as
       recall-sidecar KV)

    Args:
        video_mask (Tensor): (B, L), True=Video Token, False=Text/Pad
        attention_mask (Tensor): (B, L), 1=Valid, 0=Padding
        window_size_n (int): Number of video blocks to attend to (current + history)
        recall_video_mask (Tensor): (B, L), True for recalled-frame video tokens.
        recall_kv_mask (Tensor): (B, L), True for any ephemeral recall evidence
            tokens. Defaults to recall_video_mask.
    """
    B_limit = video_mask.shape[0]
    L_limit = video_mask.shape[1]
    if recall_video_mask is None:
        recall_video_mask = torch.zeros_like(video_mask, dtype=torch.bool)
    else:
        recall_video_mask = recall_video_mask.to(
            device=video_mask.device, dtype=torch.bool
        )
    if recall_kv_mask is None:
        recall_kv_mask = recall_video_mask
    else:
        recall_kv_mask = recall_kv_mask.to(device=video_mask.device, dtype=torch.bool)
    ordinary_video_mask = video_mask & (~recall_video_mask)

    # -------------------------------------------------------
    # 1. Pre-compute Block IDs
    # -------------------------------------------------------
    # Detect ordinary-video block starts. Recall visual blocks are sidecars:
    # they can be attended by the immediate answer turn, but they must not
    # advance or evict the ordinary visual-window block index.
    shifted = F.pad(ordinary_video_mask[:, :-1], (1, 0), value=False)
    shifted = shifted.contiguous()
    block_starts = ordinary_video_mask & (~shifted)

    # Assign IDs: (B, L)
    block_ids = block_starts.long().cumsum(dim=-1)

    # -------------------------------------------------------
    # 2. Define mask_mod (Compiled into CUDA Kernel)
    # -------------------------------------------------------
    def sliding_window_mod(b, h, q_idx, kv_idx):
        b_c = torch.clamp(b, 0, B_limit - 1)
        q_idx_c = torch.clamp(q_idx, 0, L_limit - 1)
        kv_idx_c = torch.clamp(kv_idx, 0, L_limit - 1)
        in_bounds = (b < B_limit) & (q_idx < L_limit) & (kv_idx < L_limit)

        q_is_valid = attention_mask[b_c, q_idx_c] > 0
        k_is_valid = attention_mask[b_c, kv_idx_c] > 0
        is_valid_pair = in_bounds & q_is_valid & k_is_valid

        # --- 4. Causal Constraint ---
        is_causal = q_idx_c >= kv_idx_c

        # --- 5. Content-Specific Logic ---
        k_is_video = video_mask[b_c, kv_idx_c]
        k_is_recall_kv = recall_kv_mask[b_c, kv_idx_c]
        k_is_ordinary_video = k_is_video & (~recall_video_mask[b_c, kv_idx_c])
        q_block = block_ids[b_c, q_idx_c]
        k_block = block_ids[b_c, kv_idx_c]
        diff = q_block - k_block

        # Ordinary video follows the sliding window. Recall-sidecar evidence is
        # visible only until the next ordinary video block arrives, matching
        # runtime deletion after the post-recall answer.
        ordinary_is_in_window = (~k_is_ordinary_video) | (diff < window_size_n)
        recall_is_visible = (~k_is_recall_kv) | (q_block == k_block)
        is_in_window = ordinary_is_in_window & recall_is_visible

        # --- 6. Final Combination ---
        return is_valid_pair & is_causal & is_in_window

    return sliding_window_mod


def create_mask(
    mask_mod,
    training: bool,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device: str,
):
    if torch.compiler.is_compiling():
        create_block_mask_func = create_block_mask
    else:
        create_block_mask_func = torch.compile(create_block_mask)
    return create_block_mask_func(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )


class _BaseCompiledSingleton:
    def __init_subclass__(cls, *, target_fn, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._instance = None
        cls._is_compiled = False
        cls._compiled_fn = None
        cls._target_fn = staticmethod(target_fn)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self, training=True):
        if not self._is_compiled or training != self.training:
            self.training = training
            if self._target_fn is None:
                raise ValueError(
                    f"Class {self.__class__.__name__} has no target_fn defined."
                )
            self._compiled_fn = torch.compile(self._target_fn)
            self._is_compiled = True

    def __call__(self):
        if self._compiled_fn is None:
            raise RuntimeError(f"{self._target_fn.__name__} is not compiled.")
        return self._compiled_fn


class WrappedFlexAttention(_BaseCompiledSingleton, target_fn=flex_attention):
    pass


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,  # [B, H, L, D]
    key: torch.Tensor,
    value: torch.Tensor,
    *args,
    video_block_mask: BlockMask = None,
    **kwargs,
):
    if video_block_mask is None:
        module_name = module.__class__.__name__
        if "Vision" not in module_name:
            raise TypeError(
                "streaming_attention requires video_block_mask for non-vision "
                f"attention modules; got {module_name} without one"
            )
        # Defensive fallback for old checkpoints/runs where the top-level
        # attention implementation is still propagated into the vision tower.
        # Vision attention has no ThinkStream video sliding-window mask.
        attention_mask = args[0] if args else kwargs.get("attention_mask")
        dropout_p = float(kwargs.get("dropout", 0.0) or 0.0)
        scaling = kwargs.get("scaling", None)
        is_causal = bool(kwargs.get("is_causal", False))
        try:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p if module.training else 0.0,
                is_causal=is_causal if attention_mask is None else False,
                scale=scaling,
                enable_gqa=True,
            )
        except TypeError:
            q_heads = query.shape[1]
            k_heads = key.shape[1]
            if k_heads != q_heads:
                repeat = q_heads // k_heads
                key = key.repeat_interleave(repeat, dim=1)
                value = value.repeat_interleave(repeat, dim=1)
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout_p if module.training else 0.0,
                is_causal=is_causal if attention_mask is None else False,
                scale=scaling,
            )
        return attn_output.transpose(1, 2).contiguous(), None

    if torch.compiler.is_compiling():
        flex_attn_func = flex_attention
    else:
        flex_attn_func = WrappedFlexAttention(module.training)()

    attn_output = flex_attn_func(
        query, key, value, block_mask=video_block_mask, enable_gqa=True
    )
    return attn_output.transpose(1, 2).contiguous(), None  # [B, L, H, D]


def register_streaming_attention():
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register("streaming_attention", flex_attention_forward)
