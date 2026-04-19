import re

STREAM_COLD_START = {
    "annotation_path": "/home/tione/notebook/gaozhenkun/hzh/data/ThinkStream/streaming_cot_cold_processed_5_20.jsonl",
    "data_path": "./",
}

STREAM_RLVR = {
    "annotation_path": "/home/tione/notebook/gaozhenkun/hzh/data/ThinkStream/streaming_rlvr_processed.jsonl",
    "data_path": "./",
}

# --- Agent 3-action protocol datasets (Stage 6 output) ---
STREAM_AGENT_SFT_A = {
    "annotation_path": "/home/tione/notebook/gaozhenkun/hzh/data/ThinkStream/agent/sft_final/sft_v0.1.jsonl",
    "data_path": "./",
}

STREAM_AGENT_SFT_B = {
    "annotation_path": "/home/tione/notebook/gaozhenkun/hzh/data/ThinkStream/agent/sft_final/sft_v0.1.jsonl",
    "data_path": "./",
}

STREAM_AGENT_RL = {
    "annotation_path": "/home/tione/notebook/gaozhenkun/hzh/data/ThinkStream/agent/rl_pool.jsonl",
    "data_path": "./",
}

data_dict = {
    "stream_cold_start": STREAM_COLD_START,
    "stream_rlvr": STREAM_RLVR,
    "stream_agent_sft_a": STREAM_AGENT_SFT_A,
    "stream_agent_sft_b": STREAM_AGENT_SFT_B,
    "stream_agent_rl": STREAM_AGENT_RL,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list
