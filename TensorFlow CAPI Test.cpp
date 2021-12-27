# define _CRT_SECURE_NO_WARNINGS

#include <vector>
#include <iostream>
#include <algorithm>
#include "c_api.h"
#include <opencv2/opencv.hpp>

using namespace cv;

// TF_BUFFER释放内存
static void DeallocateBuffer(void* data, size_t) {
    std::free(data);
}

// 将.pb模型文件读取到TF_BUFFER中
static TF_Buffer* ReadBufferFromFile(const char* file) {
    const auto f = std::fopen(file, "rb");
    if (f == nullptr) {
        return nullptr;
    }

    std::fseek(f, 0, SEEK_END);
    const auto fsize = ftell(f);
    std::fseek(f, 0, SEEK_SET);

    if (fsize < 1) {
        std::fclose(f);
        return nullptr;
    }

    const auto data = std::malloc(fsize);
    std::fread(data, fsize, 1, f);
    std::fclose(f);

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = DeallocateBuffer;

    return buf;
}


//从.pb模型中读取GraphDef并转化为Graph
TF_Graph* LoadGraphDef(const char* file) {
    if (file == nullptr) {
        return nullptr;
    }

    TF_Buffer* buffer = ReadBufferFromFile(file);
    if (buffer == nullptr) {
        return nullptr;
    }

    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

    TF_GraphImportGraphDef(graph, buffer, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);

    if (TF_GetCode(status) != TF_OK) {
        TF_DeleteGraph(graph);
        graph = nullptr;
    }

    TF_DeleteStatus(status);

    return graph;
}

//使用传入的Graph创建Session
TF_Session* CreateSession(TF_Graph* graph) {
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* options = TF_NewSessionOptions();
    TF_Session* sess = TF_NewSession(graph, options, status);
    TF_DeleteSessionOptions(options);

    if (TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        return nullptr;
    }

    return sess;
}

//关闭并销毁Session
bool CloseAndDeleteSession(TF_Session* sess) {
    TF_Status* status = TF_NewStatus();
    TF_CloseSession(sess, status);
    if (TF_GetCode(status) != TF_OK) {
        TF_CloseSession(sess, status);
        TF_DeleteSession(sess, status);
        TF_DeleteStatus(status);
        return false;
    }

    TF_DeleteSession(sess, status);
    if (TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        return false;
    }

    TF_DeleteStatus(status);

    return true;
}

//运行Session
//TF_Tensor包含Dtype, TensorShape和 储存Value的Buffer指针
//TF_Output包含Op和Index，是Node
bool RunSession(TF_Session* sess, const TF_Output* inputs, TF_Tensor* const* input_tensors, std::size_t ninputs,
                                                    const TF_Output* outputs, TF_Tensor** output_tensors, std::size_t noutputs) {

    if (sess == nullptr ||
        inputs == nullptr || input_tensors == nullptr ||
        outputs == nullptr || output_tensors == nullptr) {
        return false;
    }

    TF_Status* status = TF_NewStatus();

    TF_SessionRun(sess,
                           nullptr, // Run options.
                           inputs, input_tensors, static_cast<int>(ninputs), // Input tensors, input tensor values, number of inputs.
                           outputs, output_tensors, static_cast<int>(noutputs), // Output tensors, output tensor values, number of outputs.
                           nullptr, 0, // Target operations, number of targets.
                           nullptr, // Run metadata.
                           status // Output status.
    );

    if (TF_GetCode(status) != TF_OK) {  
        TF_DeleteStatus(status);
        return false;
    }

    TF_DeleteStatus(status);
    return true;
}

bool RunSession(TF_Session* sess, const std::vector<TF_Output>& inputs, const std::vector<TF_Tensor*>& input_tensors,
                                                    const std::vector<TF_Output>& outputs, std::vector<TF_Tensor*>& output_tensors) {

    return RunSession(sess, inputs.data(), input_tensors.data(), input_tensors.size(),
                                outputs.data(), output_tensors.data(), output_tensors.size());
}

//创建Tensor
TF_Tensor* CreateTensor(TF_DataType data_type, const std::int64_t* dims,
                                     std::size_t num_dims, const void* data, std::size_t len) {

    if (dims == nullptr || data == nullptr) {
        return nullptr;
    }

    TF_Tensor* tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
    if (tensor == nullptr) {
        return nullptr;
    }

    void* tensor_data = TF_TensorData(tensor);
    if (tensor_data == nullptr) {
        TF_DeleteTensor(tensor);
        return nullptr;
    }

    std::memcpy(tensor_data, data, std::min(len, TF_TensorByteSize(tensor)));

    return tensor;
}


template <typename T>
static TF_Tensor* CreateTensor(TF_DataType data_type,
                                              const std::vector<std::int64_t>& dims,
                                              const std::vector<T>& data) {

    return CreateTensor(data_type,
        dims.data(), dims.size(),
        data.data(), data.size() * sizeof(T));
}
//销毁Tensor
void DeleteTensor(TF_Tensor* tensor) {
    if (tensor == nullptr) {
        return;
    }
    TF_DeleteTensor(tensor);
}

//销毁一组Tensor
void DeleteTensors(const std::vector<TF_Tensor*>& tensors) {
    for (auto t : tensors) {
        TF_DeleteTensor(t);
    }
}


template <typename T>
static std::vector<T> GetTensorData(const TF_Tensor* tensor) {
    const auto data = static_cast<T*>(TF_TensorData(tensor));
    if (data == nullptr) {
        return {};
    }

    return { data, data + (TF_TensorByteSize(tensor) / TF_DataTypeSize(TF_TensorType(tensor))) };
}


Mat PrepareImage(const char* image_file_name) {

    Mat resized_image, float_image;
    Mat image = imread(image_file_name, 0);
    resize(image, resized_image, Size(512, 256));
    resized_image.convertTo(float_image, CV_32FC1, 1 / 255.0, 0.0);

    return float_image;
}


int main(int argc, char* argv[])
{

    std::string model_file = "./st2_coil_model.pb";                  // .pb模型路径
    TF_Graph* graph = LoadGraphDef(model_file.c_str());      //读取pb模型并创建Graph
    TF_Session* session = CreateSession(graph);                   // 使用Graph创建Session

    std::string image_name = "./39.BMP";     
    Mat input_image = PrepareImage(image_name.c_str());             //读取&预处理图片

    std::string input_node_name = "image_input";                            //输入节点名
    std::string output_node_name_1 = "mask_out1";
    std::string output_node_name_2 = "decision_out";                          //输出节点名           
    const std::vector<TF_Output> input_nodes = { { TF_GraphOperationByName(graph, input_node_name.c_str()), 0 } };     
    const std::vector<TF_Output> output_nodes = { { TF_GraphOperationByName(graph, output_node_name_1.c_str()), 0 },
                                                                            { TF_GraphOperationByName(graph, output_node_name_2.c_str()), 0 } };         //创建输入输出nodes

    const std::vector<std::int64_t> input_dims = {1, 256, 512, 1};    // 创建InputTensor
    size_t input_size = input_dims[1] * input_dims[2] * sizeof(TF_FLOAT);
    const std::vector<TF_Tensor*> input_tensors = { CreateTensor(TF_FLOAT, input_dims.data(), input_dims.size(), input_image.data,  input_size) };    //创建输入Tensor
    std::vector<TF_Tensor*> output_tensors = { nullptr,  nullptr };       //创建输出Tensor

    //Mat m = Mat(256, 512, 5, (float*)TF_TensorData(input_tensors[0]));
    //imshow("mask", m);
    //waitKey();
    bool status = RunSession(session, input_nodes, input_tensors, output_nodes, output_tensors);        //执行计算


    float* mask = static_cast<float*>(TF_TensorData(output_tensors[0]));
    std::vector<float> decision = GetTensorData<float>(output_tensors[1]);      //输出decision
    cv::Mat mask_out = cv::Mat(256, 512, 5, mask);                                       //输出mask

    imwrite("123.bmp", mask_out);
    imshow("mask", mask_out);
    waitKey();
    DeleteTensors(input_tensors);
    DeleteTensors(output_tensors);
    CloseAndDeleteSession(session);

    return 0;
}

