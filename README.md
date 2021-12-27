# TensorFlow-C_API
使用TensorFlow C_API部署pb模型

C++可以直接调用tensorflow/tensorflow/c/c_api.h中提供的函数，完成创建session，读取graph，运行session等操作。具体操作如下

1.将c_api.h添加到项目引用(include)中，并把其依赖的一系列tensorflow/tensorflow/c中的头文件添加到项目include
2.将编译得到的tensorflow.lib添加到链接器/附加依赖项，tensorflow.dll拷贝至输出目录下，即可调用c_api中的函数。

如果没有编译的tensorflow.lib和tensorflow.dll

2.将tensorflow/python/_pywrap_tensorflow_internal.lib添加到项目链接器/附加依赖项中

3.将tensorflow/python/_pywrap_tensorflow_internal.pyd 和python37.dll添加到项目生成的exe文件夹下。_pywrap_tensorflow_internal.pyd是bazel编译生成的动态链接库文件（类似dll 和 so）

tensorflow的python前端调用的就是_pywrap_tensorflow_internal.pyd中的函数

_pywrap_tensorflow_internal.lib 和_pywrap_tensorflow_internal.pyd 在使用pip install安装tensorflow完成后会生成


测试时间发现，使用_pywrap_tensorflow_internal.pyd居然比使用tensorflow.dll要快3倍以上，目前没想明白原因。
