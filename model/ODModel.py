import torch
import torch.nn as nn
import numpy as np


class ODModel(nn.Module):
    def __init__(self, N, temp, freq):
        super(ODModel, self).__init__()

        self.N = N  # 区域数
        self.temp = temp  # 频域差距 [N,]
        self.freq = freq  # 频域差距 [N,]
        n1 = 128  # 隐藏层神经元1
        n2 = 64  # 隐藏层神经元2

        # 权重[α1,α2,α3] --> shape [N, 3]
        self.weights = nn.Parameter(torch.randn(N, 3))

        # MLP层 input [B,N] --> output [B,N*N]
        self.mlp = nn.Sequential(
            nn.Linear(N, n1),
            nn.ReLU(),
            nn.Linear(n1, n2),
            nn.ReLU(),
            nn.Linear(n2, N * N)
        )

    def forward(self, x):
        '''
        前向传播
        :param x: Batch输入Speed --> [B,N]
        :return: Batch输出OD     --> [B,N,N]
        '''
        B,N = x.shape  # [B,N]

        x_temp = np.tile(self.temp, (B, 1))  # [B,N]
        x_freq = np.tile(self.freq, (B, 1))  # [B,N]

        x_random = np.random.normal(loc=0.05, scale=0.01, size=(B, N)).astype(float)  # [B,N]
        x_random = np.clip(x_random, 0, 0.1)

        x_temp_freq_rand = np.stack([x_temp, x_freq, x_random], axis=-1)  # [B,N,3]
        tensor_delta_x = torch.tensor(x_temp_freq_rand,dtype=torch.float).to(x.device)

        self.weights = self.weights.to(x.device)
        weighted_sum = x + torch.sum(tensor_delta_x * self.weights.unsqueeze(0), dim=2)

        od_matrix_flat = self.mlp(weighted_sum)  # [B,N] --> [B, N * N]
        od_matrix = od_matrix_flat.view(B, N, N)  # [B, N, N]

        return od_matrix