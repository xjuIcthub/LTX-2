# LTX-2 fork 重构计划

## L0：fork 基线（已完成）

- [x] 转移到 `xjuIcthub/LTX-2`。
- [x] 保持 `Lightricks/LTX-2` fork 关系。
- [x] 本地 origin 指向组织，upstream 指向官方。
- [x] 当前 commit 与 LTX-Desktop 固定推理 revision 对齐。

## L1：patch inventory

- 建立 ICTHub patch 清单：commit、原因、测试、上游 PR、删除条件。
- 没有本地 patch 时只 fast-forward upstream。
- 不把 FastAPI、scheduler、Redis、产品 DTO 或 artifact service 加入 fork。

## L2：runtime manifest

`resources/runtime-manifest.example.yaml` 固定 commit、checkpoint、upsampler、Gemma、LoRA hash、quantization、offload、dtype、attention、compile、CUDA/PyTorch/driver。`gpu-server` release 必须复制并填充该 manifest。

## L3：下层 adapter

`resources/adapter_contract.py.txt` 描述最小 engine Protocol。请求使用模型术语，产品标签在上层转换。输出保留 media 对象/effective parameters，编码和 artifact 上传由独立 service 完成。

可信阶段：model loading、prompt encoding、conditioning、stage 1、upsampling、stage 2、decode、encode。没有官方 callback 时 progress 为 null，cancel 只在真实停止后成为 cancelled。

## L4：验证

每次上游同步：

- package import/type/test；
- Distilled Fast 最小真实推理；
- HQ 两阶段推理；
- image conditioning、8K+1 frames、64 对齐；
- 模型 hash、VRAM、输出 manifest；
- worker 强制退出后的 CUDA/VRAM 清理。

## L5：许可证发布门

- 保留根 LICENSE 和修改声明；
- 核对收入门槛、竞争产品、机器生成内容披露和再分发义务；
- 产品 NOTICE 不沿用未复核的 Apache 分类；
- 权重与代码许可证分别记录。

## 完成门

- fork 可持续同步且 patch 最小；
- gpu-server 只按不可变 manifest 使用模型；
- Oneiroi 不再携带完整嵌套 LTX checkout；
- 发布前许可证清单通过审核。
