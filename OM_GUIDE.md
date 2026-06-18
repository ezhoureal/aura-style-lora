# 模型入端流程指南

本文档介绍模型从 PyTorch 到最终部署格式的完整转换流程。

## 概述

```
PyTorch (.pt/.pth) → ONNX (.onnx) → OMC (.omc)
```

**ONNX (Open Neural Network Exchange)** 是一个开放的模型格式，用于在不同深度学习框架之间交换模型。

**核心概念：**
- **计算图**：ONNX 使用有向无环图 (DAG) 表示模型，每个节点是一个算子 (Operator)
- **算子集 (Opset)**：不同版本的 ONNX 支持不同的算子，通过 opset version 标识
- **Proto 文件**：ONNX 模型使用 Protocol Buffers 格式存储，包含 `ModelProto`、`GraphProto`、`NodeProto` 等结构

**ONNX 文件结构：**
```
model.onnx
├── IR version          # ONNX 规范版本
├── Opset import        # 使用的算子集版本
├── Producer            # 模型生产者信息
└── Graph
    ├── Input/Output   # 输入输出张量定义
    ├── Initializer    # 常量权重
    └── Node[]         # 计算节点列表
```

**常见 opset 版本：**
| 版本 | 特性 | 推荐场景 |
|------|------|----------|
| opset 11 | 稳定，广泛支持 | 通用场景 |
| opset 13 | 改进动态 shape | 需要动态尺寸 |
| opset 17 | 最新算子支持 | 新型算子需求 |
| opset 18+ | 增强量化支持 | 量化部署 |

### 4.4 基础导出

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
model.eval()

dummy_input = torch.randn(1, 128)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
```

### 4.5 注意事项

| 问题 | 说明 | 解决方案 |
|------|------|----------|
| **动态尺寸** | 批量大小不固定 | 使用 `dynamic_axes` 或 `dynamic_shapes` |
| **算子支持** | 部分算子不支持 | 检查 ONNX 算子集，替换不支持算子 |
| **控制流** | if/loop 等需特殊处理 | 使用 `torch.jit.script` 或手动展开 |
| **数据类型** | 非 FP32 类型 | 确认目标推理引擎支持情况 |
| **opset 版本** | 不同版本算子不同 | 建议使用 opset 11/13/17 等稳定版本 |

### 4.6 常见问题

```python
# 问题1: 不支持的操作
# 解决: 替换为等价操作
x = x[x != 0]  # 不支持 → 使用 torch.masked_select + 后续处理

# 问题2: 模型包含控制流
# 解决: 使用 torch.jit.script
@torch.jit.script
def traced_model(x):
    return model(x)

# 问题3: 验证导出结果
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

---

## 2. ONNX → OMC (昇腾 NPU)

### 2.1 DDK 工具链介绍

DDK (Huawei Device Development Kit) 是华为昇腾平台的模型转换工具包，在极速空间中下载后需要解压使用。

**DDK 工具目录结构：**
```
ddk/
├── tools_ascend/    # 昇腾优化插件
├── tools_omg/       # 模型转换工具 (OM Converter)
└── tools_sysdbg/    # 模型调测工具
```

**各组件功能：**

| 组件 | 功能 | 说明 |
|------|------|------|
| `tools_ascend` | 昇腾优化插件 | 安装后使部分不支持的算子也能跑在 NPU 上，扩展算子兼容性 |
| `tools_omg` | 模型转换工具 | 将 ONNX 模型转换为昇腾适配的 OM 格式 |
| `tools_sysdbg` | 模型调测工具 | 用于模型调试、性能分析、精度验证 |

### 2.2 基础转换

```bash
# 方式1: 使用转换工具
converter --input model.onnx --output model.omc

# 方式2: Python API
from converter import OMCConverter

converter = OMCConverter()
converter.convert("model.onnx", "model.omc")
```

### 2.3 OM/OMC 格式说明

OM (Offline Model) 是昇腾 NPU 的离线模型格式，针对华为昇腾 AI 处理器优化。转换后的 OMC 模型端侧运行精度为 **FP16**。

**模型可视化与节点分析：**

OMC 模型可在 `hi-lake.rnd.huawei.com/index.html#/netron` 查看网络结构。

**节点划分规则：**
- OMC 根目录下包含多个节点时，说明模型部分算子在 NPU 执行，部分算子在 CPU 执行
- OMC 根目录下只有一个节点时，说明模型全部在 NPU 或全部在 CPU 执行

**转换流程（极速空间场景）：**

```bash
# 1. 在极速空间上下载并解压 DDK 工具包
unzip ddk_xxx.zip -d /path/to/ddk

# 2. 安装昇腾优化插件（扩展算子支持）
cd /path/to/ddk/tools_ascend
./install.sh

# 3. 使用 OMG 工具转换 ONNX → OM
cd /path/to/ddk/tools_omg
./om_converter.sh \
    --model-file=/path/to/model.onnx \
    --framework=5 \
    --output=/path/to/model \
    --input_format=NCHW \
    --input_shape="input:1,3,224,224" \
    --precision_mode=allow_fp32_to_fp16

# 4. 使用 sysdbg 进行模型调测（如需要）
cd /path/to/ddk/tools_sysdbg
./run_debug.sh --model=/path/to/model.om
```

### 2.4 常用转换参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model-file` | 输入模型路径 | `/path/to/model.onnx` |
| `--framework` | 源框架类型 (5=ONNX) | `5` |
| `--output` | 输出模型路径 | `/path/to/model` |
| `--input_format` | 输入数据格式 | `NCHW` / `NHWC` |
| `--input_shape` | 输入张量形状 | `"input:1,3,224,224"` |
| `--precision_mode` | 精度模式 | `allow_fp32_to_fp16` / `must_keep_origin_dtype` |
| `--op_select_impl_mode` | 算子实现模式 | `high_performance` / `high_precision` |

### 2.5 注意事项

| 问题 | 说明 | 解决方案 |
|------|------|----------|
| **算子兼容** | 部分 ONNX 算子不支持 | 确认 OMC 算子集，使用兼容等价实现 |
| **输入形状** | 需明确指定输入尺寸 | 通过配置文件或命令行指定 |
| **精度问题** | 量化可能导致精度损失 | 适当调整量化策略 |
| **优化选项** | 不同优化级别效果不同 | 根据性能/精度需求选择 |

### 2.6 常见算子问题

在 ONNX → OMC 转换过程中，以下算子经常出现问题：

**1. GN (Group Normalization) 算子**

| 问题 | 描述 | 影响 |
|------|------|------|
| **平台支持不备** | 部分芯片平台对 GN 算子支持不完备 | 转换失败或执行效率很低 |
| **早期尝试** | 曾尝试使用 IN (InstanceNorm) 等算子替换 GN 重新训练 | 训练效果不佳 |
| **最终方案** | 依靠海思团队完善了 GN 算子的适配 | 问题解决 |

**2. 倒数运算 (Reciprocal/Div)**

| 问题 | 描述 | 影响 |
|------|------|------|
| **精度放大** | 倒数运算会放大 FP16 带来的误差问题 | 精度损失难以完全解决 |
| **建议** | 尽量避免使用倒数运算，或使用高精度推理 | 需在训练阶段注意 |

**3. 大尺度卷积 (16×16 Conv)**

| 问题 | 描述 | 影响 |
|------|------|------|
| **平台限制** | 部分芯片不支持 16×16 的大尺度卷积 | 算子会放在 CPU 执行 |
| **影响** | 显著影响运行效率 | 需拆分或替换算子 |

**问题算子总结表：**

| 算子类型 | 问题严重程度 | 处理建议 |
|----------|-------------|----------|
| GN (GroupNorm) | ★★★★☆ | 依赖海思团队适配 |
| 倒数运算 | ★★★★★ | 尽量避免，训练时注意 |
| 16×16 Conv | ★★★☆☆ | 拆分或替换为小尺度卷积 |

### 2.6 高级配置

```python
# 昇腾优化配置示例
converter = OMCConverter(
    input_shapes={"input": [1, 3, 224, 224]},
    optimization_level=3,
    precision_mode="allow_fp32_to_fp16",  # 精度模式
    target_hardware="ascend",             # 昇腾 NPU
    insert_check_op=True,                # 插入校验算子
    output_dir="/path/to/output"
)
converter.convert("model.onnx", "model.om")
```

### 2.7 算子兼容性处理

当遇到不支持的算子时：

1. **使用昇腾优化插件**：`tools_ascend` 可将部分不支持算子替换为昇腾兼容实现
2. **手动替换算子**：将不支持的算子改写为等价的支持形式
3. **使用 TBE 自定义算子**：编写昇腾 Tensor Binary Engine 自定义算子

```python
# 常见算子替换示例
# 不支持: torch.nn.functional.interpolate (某些模式)
# 替代: 使用 Upsample + Pad 的组合

# 不支持: torch.gather (某些索引模式)
# 替代: 使用索引展开 + 选择操作
```

---

## 3. 数值精度验证

> **重要**：PyTorch → ONNX → OMC 转换过程中，**每个环节都必须验证数值精度**，确保转换后模型输出与 PyTorch 原始输出一致。

### 3.1 PyTorch → ONNX 验证

```python
import numpy as np

# 生成测试数据
test_input = torch.randn(4, 128)

# 1. PyTorch 输出
torch_model.eval()
with torch.no_grad():
    torch_output = torch_model(test_input).numpy()

# 2. ONNX 输出验证
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
onnx_output = session.run(None, {"input": test_input.numpy()})[0]

# 3. OMC 输出验证
omc_output = omc_infer("model.omc", test_input.numpy())

# 数值精度检查
onnx_diff = np.max(np.abs(torch_output - onnx_output))
omc_diff = np.max(np.abs(torch_output - omc_output))

print(f"ONNX vs PyTorch max diff: {onnx_diff:.6f}")
print(f"OMC vs PyTorch max diff: {omc_diff:.6f}")

# 阈值判断（可根据模型容忍度调整）
assert onnx_diff < 1e-5, "ONNX precision check FAILED"
assert omc_diff < 1e-3, "OMC precision check FAILED"  # FP16 允许更大误差
```

### 3.2 性能基准测试

```python
import time

def benchmark(model_path, input_data, n_runs=100):
    # 预热
    for _ in range(10):
        infer(model_path, input_data)
    
    # 计时
    start = time.time()
    for _ in range(n_runs):
        infer(model_path, input_data)
    elapsed = (time.time() - start) / n_runs
    
    return elapsed * 1000  # ms
```

---

## 4. 完整流程示例

```python
import torch
import onnx
import numpy as np
from converter import OMCConverter

# 1. 定义模型
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(16, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 2. 导出 ONNX
model = MyModel()
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=13
)

# 3. 验证 ONNX 模型格式
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model valid")

# 4. 生成测试数据并进行 PyTorch → ONNX 数值验证
test_input = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    torch_output = model(test_input).numpy()

import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
onnx_output = session.run(None, {"input": test_input.numpy()})[0]
onnx_diff = np.max(np.abs(torch_output - onnx_output))
print(f"PyTorch → ONNX max diff: {onnx_diff:.6f}")
assert onnx_diff < 1e-5, "ONNX precision check FAILED"

# 5. 转换为 OMC
converter = OMCConverter(
    input_shapes={"input": [1, 3, 224, 224]},
    optimization_level=2
)
converter.convert("model.onnx", "model.omc")

# 6. OMC 数值验证（与 PyTorch 原始输出对比）
omc_output = omc_infer("model.omc", test_input.numpy())
omc_diff = np.max(np.abs(torch_output - omc_output))
print(f"PyTorch → OMC max diff: {omc_diff:.6f} (FP16 精度)")
assert omc_diff < 1e-3, "OMC precision check FAILED"

print("All precision checks passed!")
```

---

## 5. 常见问题速查

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `Unsupported operator: xxx` | OMC 不支持该算子 | 使用兼容算子替换 |
| `Shape mismatch` | 输入形状不匹配 | 明确指定 input_shapes |
| `精度严重下降` | 量化粒度过粗 | 降低量化级别或使用 FP16 |
| `onnx.checker.check_model fails` | ONNX 模型不符合规范 | 检查模型定义 |

---

## 附录

### A. 推荐 opset 版本

- **opset 11**: 稳定，广泛支持
- **opset 13**: 改进动态 shape 支持
- **opset 17**: 最新特性，需确认支持情况

### B. 调试技巧

1. 使用 `torch.onnx.export(..., verbose=True)` 查看导出细节
2. 使用 [Netron](https://netron.app/) 可视化 ONNX 模型结构
3. 使用 OMC 可视化工具 `hi-lake.rnd.huawei.com/index.html#/netron` 查看 OMC 模型节点划分
4. 使用 `onnxsim` 简化模型图结构

### C. 精度验证阈值参考

| 转换阶段 | 推荐阈值 | 说明 |
|----------|----------|------|
| PyTorch → ONNX | < 1e-5 | FP32 之间转换，误差应极小 |
| ONNX → OMC | < 1e-3 | FP32 → FP16，误差可适当放大 |
