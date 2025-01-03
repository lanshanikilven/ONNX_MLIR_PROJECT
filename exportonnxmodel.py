import torch.nn as nn
import torch
import onnxruntime

import torch.nn as nn

model = nn.Sequential(
    a = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    b = nn.BatchNorm2d(a, num_features=16),
    c = nn.ReLU(b)
)

print(model)
# 定义一个 PyTorch 张量来模拟输入数据
batch_size = 1  # 定义批处理大小
input_shape = (batch_size, 3, 224, 224)
input_data = torch.randn(input_shape)

# 将模型转换为 ONNX 格式
output_path = "resnet50.onnx"
torch.onnx.export(model, input_data, output_path,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

# 使用 ONNX 运行时加载模型
session = onnxruntime.InferenceSession(output_path)

# 定义一个 ONNX 张量来模拟输入数据
new_batch_size = 1  # 定义新的批处理大小
new_input_shape = (new_batch_size, 3, 224, 224)
new_input_data = torch.randn(new_input_shape)

# 在 ONNX 运行时中运行模型
outputs = session.run(["output"], {"input": new_input_data.numpy()})
