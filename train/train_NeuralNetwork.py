import numpy
import time

from model.NeuralNetwork import NeuralNetwork


def train_neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate, file_prefix):
    # 神经网络实例
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # 训练数据文件
    training_data_file = open('/Volumes/hgChenArc/dataset/mnist/archive/mnist_train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 训练神经网络
    start_time = time.time()
    print("训练开始{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))))
    for epoch in range(5):
        print("第{}轮训练开始".format(epoch))
        for index, record in enumerate(training_data_list):
            values = record.split(',')
            # 输入数据归一化
            inputs = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            # 目标数据归一化
            targets = numpy.zeros(output_nodes) + 0.01
            targets[int(values[0])] = 0.99
            nn.train(inputs, targets)
            # 检查是否是第 10000, 20000, 30000... 条数据
            current_count = index + 1
            # if current_count % 10000 == 0:
            #     print("训练第{}条数据".format(current_count))
            pass

    print("训练结束{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    print("训练耗时{}".format(time.time() - start_time))

    # 保存模型
    nn.save_model(file_prefix)
