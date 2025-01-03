# tt.py
import struct
import numpy as np
dec_float = 0.1
# 十进制单精度浮点转16位16进制
hexa = struct.unpack('H',struct.pack('e',dec_float))[0]
hexa = hex(hexa)
hexa = hexa[2:]
print(hexa) # 45e6
# 16位16进制转十进制单精度浮点
y = struct.pack("H",int(hexa,16))
float = np.frombuffer(y, dtype =np.float16)[0]
print(float) # 5.9

