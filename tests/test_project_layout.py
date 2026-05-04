import importlib
import json
from pathlib import Path


def test_agent_data_root_env_resolution(monkeypatch):
    monkeypatch.setenv("THINKSTREAM_DATA_ROOT", "data/agent_v5/batch_test")
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    monkeypatch.delenv("THINKSTREAM_BATCH", raising=False)

    import scripts.agent_data_v5.config as config

    config = importlib.reload(config)
    assert config.DATA_ROOT.name == "batch_test"
    assert config.FINAL_DIR == config.DATA_ROOT / "final"

    monkeypatch.delenv("THINKSTREAM_DATA_ROOT", raising=False)
    importlib.reload(config)


def test_agent_data_root_batch_shorthand(monkeypatch):
    monkeypatch.delenv("THINKSTREAM_DATA_ROOT", raising=False)
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    monkeypatch.setenv("THINKSTREAM_BATCH", "batch_short")

    import scripts.agent_data_v5.config as config

    config = importlib.reload(config)
    assert config.DATA_ROOT == config.PROJECT_ROOT / "data" / "agent_v5" / "batch_short"

    monkeypatch.delenv("THINKSTREAM_BATCH", raising=False)
    importlib.reload(config)


def test_sft_final_dir_accepts_root_or_final(monkeypatch, tmp_path):
    root = tmp_path / "batch9"
    monkeypatch.setenv("THINKSTREAM_DATA_ROOT", str(root))
    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    monkeypatch.delenv("THINKSTREAM_FINAL_DIR", raising=False)

    import thinkstream.sft.data_list as data_list

    data_list = importlib.reload(data_list)
    assert Path(data_list.DATASET_REGISTRY["stream_agent_sft"]["annotation_path"]).parent == root / "final"

    monkeypatch.delenv("THINKSTREAM_DATA_ROOT", raising=False)
    monkeypatch.setenv("AGENT_DATA_DIR", str(root / "final"))
    data_list = importlib.reload(data_list)
    assert Path(data_list.DATASET_REGISTRY["stream_agent_sft"]["annotation_path"]).parent == root / "final"

    monkeypatch.delenv("AGENT_DATA_DIR", raising=False)
    importlib.reload(data_list)


def test_pipeline_explicit_video_list_writes_batch_manifest(monkeypatch, tmp_path):
    root = tmp_path / "batch_explicit"
    video_list = tmp_path / "videos.jsonl"
    video_list.write_text(
        json.dumps({
            "video_id": "vid0",
            "video_path": "/videos/vid0.mp4",
            "duration_sec": 31.5,
            "dataset": "unit",
        }) + "\n"
    )
    monkeypatch.setenv("THINKSTREAM_DATA_ROOT", str(root))

    import scripts.agent_data_v5.config as config
    import scripts.agent_data_v5.pipeline as pipeline

    importlib.reload(config)
    pipeline = importlib.reload(pipeline)
    videos = pipeline._load_videos_jsonl(str(video_list), limit=10)
    pipeline._write_batch_manifest(videos, source=str(video_list), seed=7)

    manifest = json.loads((root / "batch_manifest.json").read_text())
    assert manifest["batch_id"] == "batch_explicit"
    assert manifest["n_videos"] == 1
    assert (root / "selected_videos.jsonl").exists()

    monkeypatch.delenv("THINKSTREAM_DATA_ROOT", raising=False)
    importlib.reload(config)
    importlib.reload(pipeline)
