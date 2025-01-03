import json
import os

import onnxruntime as ort
import cv2
import numpy as np
import torchvision.transforms as transforms
from onnxmltools.utils import float16_converter
import onnx
from onnxsim import simplify

# 输入数据
input_data = np.fromfile("bill.bin", dtype=np.float32).reshape(1,64,8,8)
with open('bill2.bin', 'wb') as a:
     input_data.tofile(a)

print(type(input_data))
print(input_data.shape)
input_data = input_data.astype(np.float16)
#input_data = input_data.transpose(2,0,1)

#input_data = input_data.astype(np.float32).reshape(1,3,224,224)
# 加载 onnx
onnx_path= 'test_net_fp16.onnx'
sess = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider']) # 'CPUExecutionProvider'
input_name = sess.get_inputs()[0].name
output_name = [output.name for output in sess.get_outputs()]

# 推理
outputs = sess.run(output_name, {input_name:input_data})
print(type(outputs[0]))
print(outputs[0].shape)
print(outputs[0].dtype)
print(outputs[0])
#output_data = outputs[0].transpose(0, 2, 3, 1)
#print(output_data)

#model_name = onnx_path.split(".")[0]
#with open(model_name + '_output_golden.bin', 'wb') as a:
     #outputs[0].tofile(a)
# outputs = [torch.Tensor(x) for x in outputs]  # 转换为 tensor
'''
onnx_model = onnx.load_model(onnx_path)
trans_model = float16_converter.convert_float_to_float16(onnx_model,keep_io_types=False)
onnx.save_model(trans_model, "test_net_fp16.onnx")
model_simp, check = simplify(trans_model)
onnx.save(model_simp, "test_net_fp16.onnx")

'''


