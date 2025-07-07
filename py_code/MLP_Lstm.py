import torch
import torch.nn as nn


class MLP_LSTM_Model(nn.Module):
    def __init__(self):
        super(MLP_LSTM_Model, self).__init__()

        # MLP部分
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),  # 输入维度为1，输出维度为16
            nn.ReLU()
        )

        # LSTM部分
        self.lstm = nn.LSTM(input_size=16, hidden_size=24, num_layers=2, batch_first=True,bidirectional=False)

        # 输出层
        self.output_layer = nn.Linear(24, 1)  # 最终输出维度为1

    def forward(self, x):
        # MLP部分
        x = self.mlp(x)  # x的形状为[batch_size, seq_len, feature] -> [1, 4, 16]
        # LSTM部分
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out形状为[1, 4, 16]
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # 形状变为[1, 16]

        # 线性层
        output = self.output_layer(last_output)  # 最终输出形状为[1, 1]

        return output


# 生成随机输入
input_tensor = torch.rand(6, 4, 1)  # 形状为[batch_size, seq_len, feature]

# 实例化模型
model = MLP_LSTM_Model()

# 前向传播
output = model(input_tensor)

# 打印输出
print("输出结果:", output)