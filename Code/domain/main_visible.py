# -- coding: utf-8 --
# @Time : 2023/11/9 18:44
# @Author : Tsubaki_01
# @File : main_visible.py

from generate_data import generate_tensor_data
from model import Model
from visualize_output import visualize_hidden_layer_output, visualize_output
from visualize_weights import visualize_weights_and_bias_through_tensorboard

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import time

#   tensorboard log路径初始化
logdir = './logs'
if os.path.isdir(logdir):
    shutil.rmtree(logdir)
writer = SummaryWriter(logdir)

#   cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   模型
classifier = Model()
classifier = classifier.to(device)

#   模型参数
lr = 0.02
criterion = nn.MSELoss()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.Adam(classifier.parameters(), lr=lr)

#   函数参数
data_samples = 1000
data_noise = 0.075
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
time1 = time.time()
epochs = 500
for epoch in range(epochs):
    predicts = classifier(train_set)
    #   梯度下降
    optimizer.zero_grad()
    train_loss = criterion(predicts, train_labels)
    train_loss.backward()
    optimizer.step()

    #   记录loss
    writer.add_scalar(f'training loss', train_loss, epoch)

    #   权重可视化
    visualize_weights_and_bias_through_tensorboard(classifier, epoch, logdir)
time2 = time.time()
#   测试
classifier.eval()
with torch.no_grad():
    test_output = classifier(test_set)
    test_accuracy = (((test_output > 0.5) == test_labels).sum() / len(test_set)).item()
    # test_accuracy = ((torch.argmax(test_output, 1) == test_labels).sum() / len(test_set)).item()

print(f'Test Accuracy: {test_accuracy * 100:.4f}')
print(f'Training Time: {(time2 - time1) :.4f} s')

#   隐藏层与最终输出可视化
visualize_hidden_layer_output(classifier, test_set)

#   测试集可视化
visualize_output(classifier, points, labels)
