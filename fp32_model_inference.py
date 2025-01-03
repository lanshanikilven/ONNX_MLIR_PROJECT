import numpy as np
import json
import os
import onnxruntime as ort
import cv2

import torchvision.transforms as transforms

transform_GY = transforms.ToTensor()
transform_BZ= transforms.Normalize(
    mean=[123.680,116.779,103.9390],# 取决于数据集
    std=[1.0,1.0,1.0]
)
transform_compose= transforms.Compose([
# 先归一化再标准化
    transform_GY ,
    transform_BZ
])

# 输入数据
#input_data = np.fromfile("input.bin", dtype=np.float32).reshape(1, 1600, 256)
input_data = cv2.imread("ILSVRC2012_val_00000001.png")
print(type(input_data))
print(input_data.shape)
input_data = input_data.astype(np.float32)
#input_data = input_data.transpose(2,0,1)

img_transform = transform_compose(input_data)
print(type(img_transform))
print(img_transform.shape)

img_after = img_transform.numpy()
input_data = np.transpose(img_after, (1, 2, 0))
#input_data = np.transpose(img_after, (2,0,1))

input_data = input_data.reshape(1,224,224,3)

#input_data = input_data.astype(np.float32).reshape(1,3,224,224)

#input_data = input_data.reshape(1,3,224,224)
# 加载 onnx
onnx_path= 'resnet_v1_50.onnx'
#onnx_path= 'ResNet50.onnx'
#onnx_path= 'resnet50.tflite'

sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']) # 'CPUExecutionProvider'
#sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider']) # 'CPUExecutionProvider'
input_name = sess.get_inputs()[0].name
output_name = [output.name for output in sess.get_outputs()]

# 推理
outputs = sess.run(output_name, {input_name:input_data})
print(type(outputs[0]))
print(outputs[0].shape)
print(outputs[0])

final_result = np.argmax(outputs[0], axis=1)
print(final_result)
#model_name = onnx_path.split(".")[0]
#with open(model_name + '_output_golden.bin', 'wb') as a:
     #outputs[0].tofile(a)
# outputs = [torch.Tensor(x) for x in outputs]  # 转换为 tensor













