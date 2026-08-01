# LTX-2 ICTHub Refactor

该目录记录 `xjuIcthub/LTX-2` fork 如何作为纯模型/推理依赖被 `gpu-server` 使用。目录不进入 `ltx-core`、`ltx-pipelines` 或 trainer 包。

- [`plan.md`](./plan.md)：上游同步、adapter、版本固定和许可证发布门。
- [`resources/`](./resources/)：runtime manifest 与下层 adapter contract 模板。

原则：平台调度、FastAPI、产品任务和 artifact 不进入模型 fork；这里只保留最小官方同步 patch 与文档资源。
