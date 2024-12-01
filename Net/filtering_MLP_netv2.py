import torch
import torch.nn as nn


class Filter_MLP(nn.Module):
    def __init__(self):
        super(Filter_MLP, self).__init__()

        self.input_size = 9
        self.hidden_size = 1024
        self.output_size = 8

        # 定义改进后的网络结构
        self.f = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_size),  # 批归一化
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.output_size),
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.f:
            if isinstance(m, nn.Linear):
                # Xavier 初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.f(x)
