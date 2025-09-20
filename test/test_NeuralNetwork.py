import os

from train.train_NeuralNetwork import *

# --- 静态超参数 ---
# 输入节点784 28*28
input_nodes = 784
# 隐藏节点500
hidden_nodes = 500
# 输出节点10
output_nodes = 10
MODEL_DIR = '../models'

# 学习率0.1
learning_rate = 0.1

# 训练，若不需要训练，可注释下行
train_neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate,'nn_weights')

def test_neural_network():
    # 实例化一个网络（参数必须与训练时一致）
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 加载保存的权重
    nn.load_weights('../models/nn_weights_wih.npy', '../models/nn_weights_who.npy')

    # 加载测试数据
    test_data_file = open('/Volumes/hgChenArc/dataset/mnist/archive/mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 测试神经网络
    # 不匹配的列表
    no_match_list = []

    # 使用 enumerate() 来同时获取记录的索引 (index) 和数据 (record)
    for index, record in enumerate(test_data_list):
        values = record.split(',')
        # 原始图片数据（归一化后的输入）
        inputs_list = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
        # 目标值（正确的数字）
        target_num = int(values[0])
        # 神经网络查询结果
        rs = nn.query(inputs_list)
        predicted_num = numpy.argmax(rs)
        # 判断是否不匹配
        if target_num != predicted_num:
            # 存储一个元组：(该记录在测试集中的索引, 原始数据行)
            no_match_list.append((index, record))
            # 打印不匹配的记录信息
            print(f"--- 不匹配：索引 {index}, 目标 {target_num}, 预测 {predicted_num}")
        pass

    # 输出不匹配的记录数量
    print(f"不匹配的记录数量: {len(no_match_list)}")

    # 正确率
    print(f"正确率: {1 - len(no_match_list) / len(test_data_list)}")


# --- 学习率搜索范围 ---
# 集中探索 0.1 到 0.2 之间的范围
LEARNING_RATES_TO_TEST = [0.001, 0.01, 0.05, 0.08, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

# 已有的结果，用于最终对比
KNOWN_RESULTS = {
}

def get_file_prefix(learning_rate):
    """根据学习率生成唯一的文件前缀"""
    # 将浮点数转换为字符串，替换小数点以确保文件名合法
    lr_str = str(learning_rate).replace('.', 'p')
    return f'nn_lr{lr_str}'


def test_neural_network2(learning_rate):
    """针对指定的学习率测试模型，并返回正确率"""

    # 获取文件名
    file_prefix = get_file_prefix(learning_rate)
    wih_path = os.path.join(MODEL_DIR, f'{file_prefix}_wih.npy')
    who_path = os.path.join(MODEL_DIR, f'{file_prefix}_who.npy')

    # 1. 实例化和加载权重
    # 注意：这里的 learning_rate 仅用于实例化，不影响查询结果，但规范要求保持一致
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    try:
        nn.load_weights(wih_path, who_path)
    except FileNotFoundError:
        print(f"找不到学习率 {learning_rate} 的模型文件。请先训练。")
        return None

    # 2. 加载测试数据 (使用 tqdm 优化进度显示)
    test_data_file = open('/Volumes/hgChenArc/dataset/mnist/archive/mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    no_match_count = 0
    total_count = len(test_data_list)

    # 3. 测试神经网络
    # 我们可以简化为只计数，不打印每一个不匹配项，以提高运行速度
    for record in test_data_list:
        values = record.split(',')
        inputs_list = (numpy.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
        target_num = int(values[0])
        rs = nn.query(inputs_list)
        predicted_num = numpy.argmax(rs)

        if target_num != predicted_num:
            no_match_count += 1

    accuracy = 1.0 - (no_match_count / total_count)
    return accuracy


def run_experiments():
    """执行所有学习率的训练和测试"""
    global_results = KNOWN_RESULTS.copy()

    print("--- 开始学习率搜索实验 ---")

    for lr in LEARNING_RATES_TO_TEST:
        print(f"\n======== 正在处理学习率: {lr} ========")
        file_prefix = get_file_prefix(lr)

        # 1. 训练模型 (这一步将保存模型到 models/{prefix}_wih.npy 等)
        # 注意：你需要确保 train_neural_network 内部已更新为动态保存文件名
        # 例如：train_neural_network(...) 内部应调用 nn.save_model(file_prefix, MODEL_DIR)
        train_neural_network(input_nodes, hidden_nodes, output_nodes, lr, file_prefix=file_prefix)

        # 2. 测试模型
        accuracy = test_neural_network2(lr)

        if accuracy is not None:
            print(f"学习率 {lr} 对应的正确率: {accuracy:.4f}")
            global_results[lr] = accuracy

    # 3. 结果汇总
    print("\n--- 实验结果汇总 ---")
    # 按正确率从高到低排序
    sorted_results = sorted(global_results.items(), key=lambda item: item[1], reverse=True)

    for lr, acc in sorted_results:
        print(
            f"学习率: {lr:<6}, 正确率: {acc:.4f}{' <--- NEW BEST' if acc == sorted_results[0][1] and lr in LEARNING_RATES_TO_TEST else ''}")

# 运行实验
# run_experiments()
test_neural_network()