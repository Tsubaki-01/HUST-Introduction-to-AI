# -- coding: utf-8 --
# @Time : 2023/11/9 14:40
# @Author : Tsubaki_01
# @File : main.py

from generate_data import generate_tensor_data
from model import Model

import torch
import torch.nn as nn
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   模型
classifier = Model()
classifier = classifier.to(device)

#   模型参数
lr = 0.02
criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(classifier.parameters(), lr=lr)

#   函数参数
data_samples = 1000
data_noise = 0.1
random_state = 2023
data_factor = 0.5

train_test_split = int(0.8 * data_samples)

#   生成数据
points, labels = generate_tensor_data(data_samples=data_samples, data_noise=data_noise,
                                      random_state=random_state, data_factor=data_factor)
# print(points.shape, labels.shape)

#   划分训练集和测试集
train_set, train_labels = points[:train_test_split], labels[:train_test_split]
train_set = train_set.to(device)
train_labels = train_labels.to(device)

test_set, test_labels = points[train_test_split:], labels[train_test_split:]
test_set = test_set.to(device)
test_labels = test_labels.to(device)
# print(test_labels.shape)

#   训练
epochs = 250
for epoch in range(epochs):
    predicts = classifier(train_set)
    #   梯度下降
    optimizer.zero_grad()
    train_loss = criterion(predicts, train_labels)
    train_loss.backward()
    optimizer.step()

    #   测试
    classifier.eval()
    with torch.no_grad():
        test_output = classifier(test_set)
        test_accuracy = (((test_output > 0.5).float() == test_labels).sum() / len(test_set)).item()
        # test_accuracy = ((torch.argmax(test_output, 1) == test_labels).sum() / len(test_set)).item()

print(test_accuracy)
