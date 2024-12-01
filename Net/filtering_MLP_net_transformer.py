import torch
import torch.nn as nn
import torch.nn.functional as F


class Filter_MLP(nn.Module):
    def __init__(self):
        super(Filter_MLP, self).__init__()

        # 输入和输出的尺寸
        self.input_size = 9  # 假设输入是一个9维向量
        self.hidden_size = 1024
        self.output_size = 8

        # Transformer 参数
        self.num_heads = 8  # 多头注意力头数
        self.num_layers = 6  # Transformer层数

        # 线性层：输入到隐藏层
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)

        # Transformer Encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,  # 输入的特征维度
            nhead=self.num_heads,  # 注意力头数
            dim_feedforward=self.hidden_size,  # 前馈网络的维度
            dropout=0.1  # Dropout率
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=self.num_layers
        )

        # MLP层
        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, self.output_size)

        # 激活函数
        self.relu = nn.ReLU()

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 输入层处理
        x = self.input_layer(x)
        x = self.relu(x)

        # Transformer处理
        # 假设输入数据的尺寸是 (batch_size, seq_length, feature_size)
        # 在这里，我们将输入扩展为序列的形式（如果本来是单一的输入向量）
        # Transformer 需要一个形状为 (seq_length, batch_size, feature_size) 的输入
        x = x.unsqueeze(0)  # 增加序列长度维度 (1, batch_size, hidden_size)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # 将 Transformer 的输出转换为 MLP 处理
        x = x.squeeze(0)  # 移除序列长度维度，形状变为 (batch_size, hidden_size)

        # MLP部分
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc_out(x)  # 输出层
        return x
