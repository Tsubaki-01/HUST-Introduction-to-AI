# -- coding: utf-8 --
# @Time : 2023/11/9 14:13
# @Author : Tsubaki_01
# @File : model.py

import torch
import torch.nn as nn
from torch import sigmoid


#   模型；两个全连接层，均采用sigmoid激活函数
#   输入tensor类型的坐标对
class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        #   第一个全连接层，2->3
        self.fc1 = nn.Linear(2, 3)
        # self.fc1 = nn.Linear(2, 4)
        #   第二个全连接层，3->1
        self.fc2 = nn.Linear(3, 1)
        # self.fc2 = nn.Linear(4, 1)
        # self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x1 = sigmoid(self.fc1(x))
        x2 = sigmoid(self.fc2(x1))

        return x2


if __name__ == '__main__':
    test = torch.tensor([1., 4.])
    model = Model()
    out = model(test)

    x = model.named_parameters()
    print(x)
    for name, param in model.named_parameters():
        print(name)
        print(param)
        if 'fc1.weight' in name:
            print(1)
    print(out)
