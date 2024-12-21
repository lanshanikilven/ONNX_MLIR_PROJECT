# ONNX_MLIR_PROJECT

ONNX文件是使用protobuf序列化后的二进制字节流，想要读取里面的信息需要使用protobuf将其反序列化为对象才行
1.首先安装protobuf-dev，在命令行中输入protoc --version检测是否安装成功
2. 下载最新的ONNX工程，用protoc onnx-ml.proto --cpp-out=./  （根据相关的proto文件生成C++类定义文件）,会在当前目录下生成对应的onnx-ml.pb.h和onnx-ml.pb.cc,然后在printonnxinfo.cpp文件中引入头文件onnx-ml.pb.h
3. 用g++/cmake， printonnxinfo.cpp  onnx-ml.pb.cc  -lprotobuf编译生成a.out，运行该可执行文件，即可查看结果
