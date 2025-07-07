""" MultiheadAttention """

import torch
import torch.nn as nn
class SimpleMLPTransformerModel(nn.Module):
    def __init__(self):
        super(SimpleMLPTransformerModel, self).__init__()

        # MLP部分
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),  # 输入维度为1，输出维度为16
            nn.ReLU(),
            nn.Linear(16, 16)  # 输出维度仍为16（为了保持4x16的形状）
        )


        self.MH = nn.MultiheadAttention(embed_dim=16, num_heads=4, dropout=0.1, batch_first=True)
        # nn.attention

        # 输出层
        self.output_layer = nn.Linear(16, 1)  # 最终输出维度为1

    def forward(self, x, x0, y0):
        # MLP部分
        x = self.mlp(x)
        x0 = self.mlp(x0)
        y0 = self.mlp(y0)
        x, _ = self.MH(x, x0, y0)
        output = self.output_layer(x)  # 最终输出形状为[batch_size, output_dim]，即[4, 1]
        return output
# 生成随机输入
input_tensor = torch.rand(4,1)  # 形状为[batch_size，input_features]
#生成实际的键和值
key=torch.rand(8,1)
value=torch.rand(8,1)
model = SimpleMLPTransformerModel()
output = model(input_tensor,key,value)