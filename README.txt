# ONNX_MLIR_PROJECT

ONNX文件是使用protobuf序列化后的二进制字节流，想要读取里面的信息需要使用protobuf将其反序列化为对象才行
1.首先安装protobuf-dev（在目录下执行./autogen.sh，然后在bashrc中加入安装路径），在命令行中输入protoc --version检测是否安装成功
2. 下载最新的ONNX工程，用protoc onnx-ml.proto --cpp-out=./  （根据相关的proto文件生成C++类定义文件）,会在当前目录下生成对应的onnx-ml.pb.h和onnx-ml.pb.cc,然后在printonnxinfo.cpp文件中引入头文件onnx-ml.pb.h
3. 用g++/cmake， printonnxinfo.cpp  onnx-ml.pb.cc  -lprotobuf编译生成a.out，运行该可执行文件，即可查看结果


profobuf使用以及ONNX模型解析
跨平台跨语言的通用序列化方法主要有4种格式：
文本格式：包括XML和json
二进制格式：protobuf和flatbuffer
而序列化是将数据转换成二进制字节流，反序列化是从二进制字节流中提取出数据。

onnx模型包含了GraphProto，而GraphProto包含了:
（1）ValueInfoProto（包含图的input和output节点），
（2）NodeProto（表示Operator，该结构体里面还包含了operator的input/output/op_type等信息），
（3）TensorProto（表示initializer，存放权重），
（4）AttributeProto（表示node的attribute，如conv的padding，stride等）

onnx helper的make_tensor_value_info（"hast"，TensorProto，shape），可以构造出inputs和outputs，
make_tensor可以构造initializer，
make_node可以构造node，然后通过node.attribute.extend([])来增加padding，stride等信息，[]中的信息可以通过make_attribute构造出来
接着可以用make_graph(【node1】，【node2】，...， "name"，inputs，outputs，initializer=)构造出graph，最后通过make_model（graph）构造出onnx模型

ONNX提供了API：onnx.checker.check_model来判断一个onnx模型是否满足标准
还提供了onnx.utils.extract_model（"原始onnx模型"，"截取的onnx模型"，[起始点张量]，[结束点张量]）可以提取onnx模型片段


