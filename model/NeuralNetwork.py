import numpy
import os
import scipy.special


class NeuralNetwork:

    # 初始化神经网络
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        初始化神经网络

        参数:
        input_nodes (int): 输入层节点数量
        hidden_nodes (int): 隐藏层节点数量
        output_nodes (int): 输出层节点数量
        learning_rate (float): 学习率，控制权重更新的步长
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 激活函数
        # lambda 表达式是一种匿名函数，它可以在需要函数对象的任何地方使用
        # 这里使用 lambda 表达式定义了一个匿名函数，它的参数是 x，返回值是 scipy.special.expit(x)
        self.activation_function = lambda x: scipy.special.expit(x)

        # 初始化权重
        # 输入层到隐藏层的权重
        # pow() 方法返回 x^y（x 的 y 次方） 的值。 pow(x,-0.5) 表示 x 的 -0.5 次方,即x的标准差的倒数
        # self.wih = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        # 权重设置根据输入节点数量和隐藏节点数量的平方根倒数
        self.wih = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # 隐藏层到输出层的权重
        # self.who = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        self.who = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        pass

    # 训练神经网络
    def train(self, inputs_list, targets_list):
        # 将输入列表转换为二维数组
        t_inputs = numpy.array(inputs_list, ndmin=2, dtype=float).T
        t_targets = numpy.array(targets_list, ndmin=2, dtype=float).T
        hidden_inputs = numpy.dot(self.wih, t_inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层的误差
        output_errors = t_targets - final_outputs
        # 计算隐藏层的误差
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 更新隐藏层到输出层的权重
        self.who += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                   hidden_outputs.T)
        # 更新输入层到隐藏层的权重
        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                   t_inputs.T)
        pass

    # 查询神经网络
    # 接收神经网络的输入，返回网络的输出
    def query(self, inputs_list):
        # 将输入列表转换为二维数组
        q_inputs = numpy.array(inputs_list, ndmin=2, dtype=float).T
        # 隐藏层输入
        hidden_inputs = numpy.dot(self.wih, q_inputs)
        # 隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层输出
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    # 保存模型：保存模型的权重到指定目录
    def save_model(self, file_prefix, model_dir='../models/'):
        """保存模型权重到指定目录。"""
        wih_path = os.path.join(model_dir, f'{file_prefix}_wih.npy')
        who_path = os.path.join(model_dir, f'{file_prefix}_who.npy')
        numpy.save(wih_path, self.wih)
        numpy.save(who_path, self.who)
        print(f"模型权重已保存到: {wih_path} 和 {who_path}")
        pass

    # 加载权重：恢复模型
    def load_weights(self, wih_path, who_path):
        """从文件中加载权重矩阵。"""
        self.wih = numpy.load(wih_path)
        self.who = numpy.load(who_path)
        print("模型权重已成功加载...")
        pass