# Task 数据与推理

每个任务目录直接放规范化的提示词 CSV；每个 CSV 对应一个同名输出目录：

```text
tasks/
└── sep14/
    ├── spatial_relationship-multi_instance.csv  # 精确表头：id,prompt
    ├── spatial_relationship-multi_instance/
    │   ├── video_000.mp4
    │   └── video_001.mp4
    └── ...
```

CSV 使用 UTF-8 编码，第一行必须是 `id,prompt`，每行必须正好两列；prompt 中的逗号由 CSV 引用处理。视频按 CSV 内行顺序命名为 `video_000.mp4`、`video_001.mp4`，视频目录与 CSV 的 stem 相同。

验证任务格式：

```bash
uv run --no-sync python tasks/validate.py sep14
uv run --no-sync python tasks/validate.py sep14 --check-videos
```

运行 HQ 推理。默认参数与 H200 上刚使用的配置一致：完整 BF16 LTX-2.3 dev 权重、HQ Res2s、15+3 步、1920×1088、24 fps、5 秒、含音频：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python tasks/infer.py sep14
```

任务会复用已完成的视频；需要重跑时加 `--overwrite`。模型权重、Gemma 和 LoRA 默认从仓库的 `models/` 读取，也可以用对应命令行参数覆盖。

多卡使用多进程，每个进程只看到并使用一张卡。前面的进程按轮转分配视频，最后一个进程扫描全部未完成视频并领取剩余任务作为兜底；一条视频完成并校验后立即原子改名为最终的 `video_00x.mp4`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync python tasks/infer.py sep14 --num-processes 4
```

`--num-processes 0` 表示使用 `CUDA_VISIBLE_DEVICES` 中的全部卡，也可以用 `--gpu-ids 0,2,3` 指定卡。单个视频失败时默认在同一轮最多重试两次（可用 `--max-retries` 调整）。

上传到 Hugging Face 数据集仓库 `xjuIcthub/tasks`：

```bash
uv run --no-sync python tasks/upload.py sep14
uv run --no-sync python tasks/upload.py --all
```

上传后的路径保持为 `sep14/<csv同名目录>/video_000.mp4`，CSV 保持为 `sep14/<文件名>.csv`。上传脚本只上传 `.csv` 和 `.mp4`，会忽略 manifest、日志和临时文件；使用 Hugging Face 缓存登录或 `HF_TOKEN`。
