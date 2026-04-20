# 数据集审计报告

**审计时间**: 2026-04-20  
**审计路径**: `/home/tione/notebook/gaozhenkun/hzh/data/datasets`

---

## 一、目录总览

| 数据集 | 磁盘占用 | 视频数量 | 状态 |
|--------|---------|---------|------|
| VideoMind-Dataset | ~10T | ~1,012,962 | 已解压，压缩包已清理 |
| tarsier2_unzip | ~4.3T | ~300,000+ | 大部分已解压，WebVid解压中 |
| LLaVA-Video-178K | ~2.4T | ~200,000+ | 已解压，压缩包已清理 |
| Koala_raw | ~891G | ~25,853 | 已解压 |
| how_to_step | ~201G | ~18,836 | 已解压 |
| how_to_caption | ~154G | 未统计 | 已解压 |
| Koala | ~134G | ~81,489 | 已解压 |
| Koala-36M | ~16K | 0 | 仅元数据 |
| **总计** | **~18T** | **~1,000,000+** | |

---

## 二、压缩包处理情况

### 已删除（解压后清理）

| 数据集 | 删除的压缩包 |
|--------|------------|
| LLaVA-Video-178K (全部子目录) | 250个 tar.gz 分片 |
| VideoMind-Dataset/cgbench | subtitles.tar.gz |
| VideoMind-Dataset/charades_sta | videos_3fps_480_noaudio.tar.gz |
| VideoMind-Dataset/didemo | videos_3fps_480_noaudio.tar.gz |
| VideoMind-Dataset/lvbench | videos_3fps_480_noaudio.tar.gz |
| VideoMind-Dataset/nextqa | videos.tar.gz |
| VideoMind-Dataset/queryd | videos_3fps_480_noaudio.tar.gz |
| VideoMind-Dataset/tacos | videos.tar.gz + videos_3fps_480_noaudio.tar.gz |
| VideoMind-Dataset/videomme | subtitles.tar.gz |
| VideoMind-Dataset/youcook2 | videos_3fps_480_noaudio.tar.gz |
| VideoMind-Dataset/mvbench | video.tar.gz (17G) |
| tarsier2_unzip/TREC-VTT | videos.tar |
| tarsier2_unzip/SSV2 | videos.tar |

### 解压中（后台运行）

| 数据集 | 方式 | 状态 |
|--------|------|------|
| tarsier2_unzip/WebVid-10M_part-1 | cat part-* \| tar -x | 后台运行中 |
| tarsier2_unzip/WebVid-10M_part-2 | cat part-* \| tar -x | 后台运行中 |
| tarsier2_unzip/WebVid-10M_part-3 | cat part-* \| tar -x | 后台运行中 |

### 仍有分片文件（已解压，分片保留）

以下数据集已有 videos 目录且有视频，但 `.tar.gz.00/.01...` 或 `.tar.part-*` 分片文件仍存在（原始下载分片，非压缩包）：

- VideoMind-Dataset: activitynet, cosmo_cap, hirest, internvid_vtime, longvideobench, mlvu, qvhighlights, vid_morp 等
- tarsier2_unzip: ActivityNet, Charades, Charades-Ego, Kinetics-700, LSMDC_part-1~4, Oops, VATEX, TGIF

> 注意：这些 `.tar.gz.00` 分片是原始下载文件，已合并解压，可按需删除以释放空间。

---

## 三、各数据集详细状态

### VideoMind-Dataset

| 子数据集 | 视频目录 | 视频数 | 备注 |
|---------|---------|-------|------|
| activitynet | videos + videos_3fps_480_noaudio | ~19,994×2 | 双版本 |
| cgbench | videos + videos_3fps_480_noaudio | ~1,219 | |
| charades_sta | videos | ~9,848 | |
| coin | videos | 未统计 | |
| cosmo_cap | videos + videos_3fps_480_noaudio | ~68,057×2 | 双版本 |
| didemo | videos | ~10,463 | |
| hirest | videos + videos_3fps_480_noaudio | 未统计 | 双版本 |
| internvid_vtime | videos + videos_crop_3fps_480_noaudio | ~45,913×2 | 双版本 |
| longvideobench | videos | 未统计 | |
| lvbench | videos | ~103 | |
| mlvu | video | 未统计 | |
| mvbench | video | ~3,321 | 多子目录结构 |
| nextqa | videos | ~159 | |
| queryd | videos | ~2,859 | |
| qvhighlights | videos + videos_3fps_480_noaudio | 未统计 | 双版本 |
| tacos | videos + videos_3fps_480_noaudio | ~608 | 双版本 |
| videomme | videos | ~900 | |
| vid_morp | videos + videos_3fps_480_noaudio | ~47,026×2 | 双版本 |
| youcook2 | videos | ~1,806 | |
| ego4d | v1/v2 | 未统计 | |

### tarsier2_unzip

| 子数据集 | 视频数 | 格式 | 状态 |
|---------|-------|------|------|
| ActivityNet | 未统计 | mp4 | 已解压 |
| Charades | 未统计 | mp4 | 已解压 |
| Charades-Ego | 未统计 | mp4 | 已解压 |
| Kinetics-700 | ~49,673 | mp4 | 已解压 |
| LSMDC_part-1~4 | ~27,000×4 | mp4 | 已解压 |
| Oops | 未统计 | mp4 | 已解压 |
| SSV2 | ~9,996 | webm | 已解压 |
| TGIF | ~94,775 | gif | 已解压 |
| TREC-VTT | 未统计 | mp4 | 已解压 |
| VATEX | ~22,422 | mp4 | 已解压 |
| WebVid-10M_part-1 | 未统计 | mp4 | **解压中** |
| WebVid-10M_part-2 | 未统计 | mp4 | **解压中** |
| WebVid-10M_part-3 | 未统计 | mp4 | **解压中** |

### LLaVA-Video-178K

所有子目录均已解压，压缩包已删除。视频按时长分类：

| 子目录 | 时长范围 | 来源 |
|--------|---------|------|
| 0_30_s_* | 0~30秒 | academic/activitynetqa/nextqa/perceptiontest/youtube |
| 30_60_s_* | 30~60秒 | academic/activitynetqa/nextqa/perceptiontest/youtube |
| 1_2_m_* | 1~2分钟 | academic/activitynetqa/nextqa/youtube |
| 2_3_m_* | 2~3分钟 | academic/activitynetqa/nextqa/youtube |

---

## 四、视频时长抽样

| 视频 | 数据集 | 时长 |
|------|--------|------|
| cosmo_cap sample | VideoMind | ~69s |
| vid_morp sample | VideoMind | ~34s |
| internvid_vtime sample | VideoMind | ~33s |
| TREC-VTT sample | tarsier2 | ~6s |
| LLaVA 0_30s sample | LLaVA-Video | ~15s |
| LLaVA 2_3m sample | LLaVA-Video | ~150s |

---

## 五、视频目录 CSV

生成中，路径：`/home/tione/notebook/gaozhenkun/hzh/data/datasets/video_catalog.csv`

字段：`video_path, filename, dataset, subdataset, size_bytes, duration_sec, width, height, fps, codec`

进度日志：`/home/tione/notebook/gaozhenkun/hzh/data/datasets/catalog.log`

---

## 六、关键发现与建议

1. **双版本冗余**：VideoMind 中多个数据集同时存有原始视频和 `3fps_480_noaudio` 处理版本，可按需删除原始版本节省约 3~4T 空间。

2. **分片文件残留**：大量 `.tar.gz.00/.01` 和 `.tar.part-*` 原始下载分片仍占用空间，已确认解压完成后可全部删除。

3. **WebVid-10M**：三个 part 正在解压，完成后需补充到 video_catalog.csv。

4. **命名不一致**：视频目录在不同数据集中命名为 `videos`、`video`、`videos_3fps_480_noaudio`、`videos_crop_3fps_480_noaudio` 等，CSV 已统一记录实际路径。
