import torch
import torch.nn as nn


class Filter_MLP(nn.Module):
    def __init__(self):
        super(Filter_MLP, self).__init__()

        self.input_size = 9
        self.hidden_size = 1024
        self.output_size = 8

        # 定义改进后的网络结构
        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.layer1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size, 512)
        self.layer4 = nn.Linear(512, 256)
        self.layer5 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, self.output_size)

        # BatchNorm层
        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.bn3 = nn.BatchNorm1d(self.hidden_size)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

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
        # 第一层输入
        residual = x
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.bn1(x)

        # 第一层残差连接
        x = self.layer1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x += residual  # 残差连接

        # 第二层残差连接
        residual = x
        x = self.layer2(x)
        x = self.relu(x)
        x = self.bn3(x)
        x += residual  # 残差连接

        # 第三层
        residual = x
        x = self.layer3(x)
        x = self.relu(x)
        x = self.bn4(x)

        # 第四层
        residual = x
        x = self.layer4(x)
        x = self.relu(x)
        x = self.bn5(x)

        # 第五层
        residual = x
        x = self.layer5(x)
        x = self.relu(x)
        x = self.bn6(x)

        # 输出层
        x = self.output_layer(x)

        return x
