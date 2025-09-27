import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class NeuralNetworkPyTorch(nn.Module):
    """
    基于PyTorch的神经网络实现
    支持GPU加速和自动微分
    """

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, device='auto'):
        """
        初始化PyTorch神经网络

        参数:
        input_nodes (int): 输入层节点数量
        hidden_nodes (int): 隐藏层节点数量
        output_nodes (int): 输出层节点数量
        learning_rate (float): 学习率
        device (str): 设备选择，'auto'自动选择，'cuda'强制GPU，'cpu'强制CPU
        """
        super(NeuralNetworkPyTorch, self).__init__()
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 定义网络层
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)
        
        # 初始化权重
        self._initialize_weights()
        
        # 将模型移动到指定设备
        self.to(self.device)
        
        # 优化器
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.MSELoss()

    def _initialize_weights(self):
        """初始化网络权重"""
        # Xavier初始化
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """前向传播"""
        # 隐藏层
        x = torch.sigmoid(self.fc1(x))
        # 输出层
        x = torch.sigmoid(self.fc2(x))
        return x

    def train_step(self, inputs, targets):
        """单步训练"""
        # 转换为张量并移动到设备
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        # 前向传播
        outputs = self.forward(inputs)
        
        # 计算损失
        loss = self.criterion(outputs, targets)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def query(self, inputs):
        """查询网络输出"""
        with torch.no_grad():
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            outputs = self.forward(inputs)
            return outputs.cpu().numpy()

    def train_batch(self, inputs_list, targets_list, batch_size=32):
        """批量训练"""
        total_loss = 0
        num_batches = len(inputs_list) // batch_size
        
        for i in range(0, len(inputs_list), batch_size):
            batch_inputs = inputs_list[i:i+batch_size]
            batch_targets = targets_list[i:i+batch_size]
            
            # 转换为张量
            batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32, device=self.device)
            batch_targets = torch.tensor(batch_targets, dtype=torch.float32, device=self.device)
            
            # 前向传播
            outputs = self.forward(batch_inputs)
            
            # 计算损失
            loss = self.criterion(outputs, batch_targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches

    def save_model(self, file_prefix, model_dir='../models/'):
        """保存模型"""
        model_path = os.path.join(model_dir, f'{file_prefix}_pytorch.pth')
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_nodes': self.input_nodes,
            'hidden_nodes': self.hidden_nodes,
            'output_nodes': self.output_nodes,
            'learning_rate': self.learning_rate,
            'device': str(self.device)
        }, model_path)
        print(f"PyTorch模型已保存到: {model_path}")

    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print("PyTorch模型已成功加载")

    def get_device_info(self):
        """获取设备信息"""
        if self.device.type == 'cuda':
            print(f"GPU设备: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        else:
            print("使用CPU设备")

    def evaluate(self, test_inputs, test_targets):
        """评估模型性能"""
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in zip(test_inputs, test_targets):
                inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
                targets = torch.tensor(targets, dtype=torch.float32, device=self.device)
                
                outputs = self.forward(inputs)
                predicted = torch.argmax(outputs)
                actual = torch.argmax(targets)
                
                if predicted == actual:
                    correct += 1
                total += 1
        
        accuracy = correct / total
        print(f"准确率: {accuracy:.4f} ({correct}/{total})")
        return accuracy


# 兼容性包装器，保持与原接口一致
class NeuralNetworkPyTorchWrapper:
    """PyTorch神经网络的兼容性包装器"""
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, device='auto'):
        self.nn = NeuralNetworkPyTorch(input_nodes, hidden_nodes, output_nodes, learning_rate, device)
    
    def train(self, inputs_list, targets_list):
        """兼容原接口的训练方法"""
        return self.nn.train_step(inputs_list, targets_list)
    
    def query(self, inputs_list):
        """兼容原接口的查询方法"""
        return self.nn.query(inputs_list)
    
    def save_model(self, file_prefix, model_dir='../models/'):
        """兼容原接口的保存方法"""
        return self.nn.save_model(file_prefix, model_dir)
    
    def load_weights(self, wih_path, who_path):
        """兼容原接口的加载方法（需要转换）"""
        print("注意：PyTorch版本使用不同的模型格式，请使用load_model方法")
    
    def get_gpu_info(self):
        """获取GPU信息"""
        return self.nn.get_device_info()
