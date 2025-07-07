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

        # Transformer部分
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)  # 2层
        self.MH=nn.MultiheadAttention(embed_dim=16, num_heads=4, dropout=0.1, batch_first=True)
        #nn.attention

        # 输出层
        self.output_layer = nn.Linear(16, 1)  # 最终输出维度为1

    def forward(self, x,x0,y0):
        # MLP部分
        x = self.mlp(x)  # x的形状为[4,4,16]
        #attention_mask,为1的位置会被允许进行注意力计算
        attention_mask1 = torch.ones(x.size(0), x.size(1), dtype=torch.float32).to(x.device)
        x =self.transformer_encoder(x, src_key_padding_mask=attention_mask1)
        # 平均池化
        x=torch.mean(x, dim=1)
        # [1, 4, 16] -> [4, 16]（保持形状）
        # 线性层
        output = self.output_layer(x)  # 最终输出形状为[batch_size, output_dim]，即[4, 1]
        return output


# 生成随机输入
input_tensor = torch.rand(4,4,1)  # 形状为[batch_size，seq, input_features]
#生成实际的键和值
key=torch.rand(4,4,1)
value=torch.rand(4,4,1)
# 实例化模型
model = SimpleMLPTransformerModel()

# 前向传播
output = model(input_tensor)

# 打印输出
print("输出结果:", output)