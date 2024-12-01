import torch
import torch.nn as nn
import torch.nn.functional as F


class Filter_MLP(nn.Module):
    def __init__(self):
        super(Filter_MLP, self).__init__()
        # 定义第一层：从输入到隐藏层
        self.input_size = 9
        self.hidden_size = 1024
        self.output_size = 8

        self.f = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, self.output_size),
        )

        # self.normalize_input = False


    def __str__(self):
        # print("Embedding_MLP", self.f)
        return "Embedding_MLP\n" + str(self.f)

    def forward(self, x):
        # 前向传播：输入 -> 隐藏层 -> 激活 -> 输出
        # if self.normalize_input:
        #     x = F.normalize(x, p=2, dim=1)

        # print(x)

        out = self.f(x)
        # out = torch.sigmoid(out)

        # print(out)
        return out
