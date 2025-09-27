import torch
import time
import numpy as np
from model.NeuralNetwork_PyTorch import NeuralNetworkPyTorch, NeuralNetworkPyTorchWrapper


def train_neural_network_pytorch(input_nodes, hidden_nodes, output_nodes, learning_rate, file_prefix, device='auto'):
    """
    PyTorch版本的神经网络训练函数
    
    参数:
    input_nodes: 输入层节点数
    hidden_nodes: 隐藏层节点数  
    output_nodes: 输出层节点数
    learning_rate: 学习率
    file_prefix: 模型保存前缀
    device: 设备选择
    """
    # 创建PyTorch神经网络实例
    nn = NeuralNetworkPyTorch(input_nodes, hidden_nodes, output_nodes, learning_rate, device)
    
    # 训练数据文件
    training_data_file = open('/Volumes/hgChenArc/dataset/mnist/archive/mnist_train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 显示设备信息
    print("=== 设备信息 ===")
    nn.get_device_info()
    print("===============")

    # 训练神经网络
    start_time = time.time()
    print("PyTorch训练开始{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))))
    
    # 准备训练数据
    print("准备训练数据...")
    inputs_list = []
    targets_list = []
    
    for record in training_data_list:
        values = record.split(',')
        # 输入数据归一化
        inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
        # 目标数据归一化
        targets = np.zeros(output_nodes) + 0.01
        targets[int(values[0])] = 0.99
        
        inputs_list.append(inputs)
        targets_list.append(targets)
    
    print(f"训练数据准备完成，共{len(inputs_list)}条数据")
    
    # 批量训练
    batch_size = 64
    num_epochs = 5
    
    for epoch in range(num_epochs):
        print(f"第{epoch+1}轮训练开始")
        epoch_start = time.time()
        
        # 随机打乱数据
        indices = np.random.permutation(len(inputs_list))
        shuffled_inputs = [inputs_list[i] for i in indices]
        shuffled_targets = [targets_list[i] for i in indices]
        
        # 批量训练
        avg_loss = nn.train_batch(shuffled_inputs, shuffled_targets, batch_size)
        
        epoch_time = time.time() - epoch_start
        print(f"第{epoch+1}轮训练完成，平均损失: {avg_loss:.6f}，耗时: {epoch_time:.2f}秒")
        
        # 显示设备信息
        if epoch % 2 == 0:  # 每两轮显示一次
            nn.get_device_info()

    total_time = time.time() - start_time
    print("PyTorch训练结束{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    print("总训练耗时{}".format(total_time))

    # 保存模型
    nn.save_model(file_prefix)
    
    # 最终设备信息
    print("\n=== 训练完成后的设备信息 ===")
    nn.get_device_info()
    
    return nn


def train_with_compatibility_wrapper(input_nodes, hidden_nodes, output_nodes, learning_rate, file_prefix):
    """
    使用兼容性包装器训练（保持与原接口一致）
    """
    # 创建兼容性包装器
    nn = NeuralNetworkPyTorchWrapper(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    # 训练数据文件
    training_data_file = open('/Volumes/hgChenArc/dataset/mnist/archive/mnist_train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 显示设备信息
    print("=== 设备信息 ===")
    nn.get_gpu_info()
    print("===============")

    # 训练神经网络
    start_time = time.time()
    print("PyTorch兼容模式训练开始{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))))
    
    for epoch in range(5):
        print("第{}轮训练开始".format(epoch))
        epoch_start = time.time()
        
        for index, record in enumerate(training_data_list):
            values = record.split(',')
            # 输入数据归一化
            inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            # 目标数据归一化
            targets = np.zeros(output_nodes) + 0.01
            targets[int(values[0])] = 0.99
            
            # 训练
            nn.train(inputs, targets)
            
            # 每10000条数据显示一次进度
            if (index + 1) % 10000 == 0:
                print(f"训练第{index + 1}条数据")

        epoch_time = time.time() - epoch_start
        print(f"第{epoch}轮训练完成，耗时: {epoch_time:.2f}秒")

    total_time = time.time() - start_time
    print("PyTorch兼容模式训练结束{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    print("总训练耗时{}".format(total_time))

    # 保存模型
    nn.save_model(file_prefix)


def compare_frameworks_performance(input_nodes=784, hidden_nodes=500, output_nodes=10, learning_rate=0.1):
    """
    比较不同框架的性能
    """
    print("=== 框架性能对比测试 ===")
    
    # 测试数据量
    test_size = 1000
    
    # 生成测试数据
    test_inputs = np.random.rand(test_size, input_nodes)
    test_targets = np.random.rand(test_size, output_nodes)
    
    # NumPy CPU测试
    print("\n--- NumPy CPU测试 ---")
    from model.NeuralNetwork import NeuralNetwork
    nn_numpy = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    start_time = time.time()
    for i in range(test_size):
        nn_numpy.train(test_inputs[i], test_targets[i])
    numpy_time = time.time() - start_time
    print(f"NumPy CPU训练{test_size}条数据耗时: {numpy_time:.4f}秒")
    
    # CuPy GPU测试
    # print("\n--- CuPy GPU测试 ---")
    # try:
    #     from model.NeuralNetwork_GPU import NeuralNetworkGPU
    #     nn_cupy = NeuralNetworkGPU(input_nodes, hidden_nodes, output_nodes, learning_rate, use_gpu=True)
    #
    #     start_time = time.time()
    #     for i in range(test_size):
    #         nn_cupy.train(test_inputs[i], test_targets[i])
    #     cupy_time = time.time() - start_time
    #     print(f"CuPy GPU训练{test_size}条数据耗时: {cupy_time:.4f}秒")
    #     print(f"CuPy加速比: {numpy_time / cupy_time:.2f}x")
    # except ImportError:
    #     print("CuPy未安装，跳过测试")
    
    # PyTorch测试
    print("\n--- PyTorch测试 ---")
    nn_pytorch = NeuralNetworkPyTorch(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    start_time = time.time()
    for i in range(test_size):
        nn_pytorch.train_step(test_inputs[i], test_targets[i])
    pytorch_time = time.time() - start_time
    print(f"PyTorch训练{test_size}条数据耗时: {pytorch_time:.4f}秒")
    print(f"PyTorch加速比: {numpy_time / pytorch_time:.2f}x")


if __name__ == "__main__":
    # 运行性能对比测试
    compare_frameworks_performance()
    
    # 如果需要训练完整模型，取消下面的注释
    # train_neural_network_pytorch(784, 500, 10, 0.1, 'nn_pytorch_weights')
    # train_with_compatibility_wrapper(784, 500, 10, 0.1, 'nn_pytorch_compat_weights')
