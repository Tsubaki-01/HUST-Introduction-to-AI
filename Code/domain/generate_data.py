# -- coding: utf-8 --
# @Time : 2023/11/9 14:06
# @Author : Tsubaki_01
# @File : generate_data.py

from sklearn.datasets import make_circles
import torch


#   生成同心圆数据，返回datas与targets的张量
def generate_tensor_data(data_samples=1000, data_noise=0.1, random_state=2023, data_factor=0.5):
    datas, targets = make_circles(n_samples=data_samples, shuffle=True,
                                  noise=data_noise, random_state=random_state, factor=data_factor)

    datas = torch.tensor(datas, dtype=torch.float32)
    targets = torch.reshape(torch.tensor(targets, dtype=torch.float32), [-1, 1])
    # targets = torch.tensor(targets, dtype=torch.long)

    return datas, targets


if __name__ == '__main__':
    x, y = generate_tensor_data()
    print(x.shape, y.shape)
