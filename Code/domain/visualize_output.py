# -- coding: utf-8 --
# @Time : 2023/11/9 19:18
# @Author : Tsubaki_01
# @File : visualize_output.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
from sklearn.svm import SVC


def visualize_hidden_layer_output(model, data):
    model.eval()

    # 获取隐藏层的输出
    with torch.no_grad():
        hidden_layer_output = model.fc1(data).sigmoid()
        hidden_layer_output = hidden_layer_output.cpu()
        final_output = model(data)
        final_output = (final_output > 0.5).float().ravel()
        color_map = ['red' if i > 0.5 else 'blue' for i in final_output]

    data = data.cpu()
    final_output = final_output.cpu()
    fig = plt.figure()

    # 使用支持向量机找到分类平面
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(hidden_layer_output, final_output)
    # 获取分类平面的参数
    w = svm.coef_[0]
    xx = np.linspace(0, 1)
    yy = np.linspace(0, 1)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (-svm.intercept_[0] - w[0] * XX - w[1] * YY) / w[2]

    ZZ_masked = np.ma.masked_where(ZZ > 1, ZZ)
    ZZ_masked = np.ma.masked_where(ZZ < 0, ZZ_masked)

    # 绘制3D图像
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=hidden_layer_output[:, 0], ys=hidden_layer_output[:, 1], zs=hidden_layer_output[:, 2],
               c=color_map)
    ax.plot_surface(XX, YY, ZZ_masked, rstride=1, cstride=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hidden Layer Output (Three Neurons -> X, Y, Z)')
    plt.show()
    # 绘制2D图像
    plt.figure(figsize=(34, 8))
    plt.subplot(1, 3, 1)
    scatter1 = plt.scatter(data[:, 0], data[:, 1], c=hidden_layer_output[:, 0], cmap='viridis', edgecolors='k')
    # 添加颜色条
    colorbar = plt.colorbar(scatter1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Hidden Layer Output visualization 1')
    plt.subplot(1, 3, 2)
    scatter2 = plt.scatter(data[:, 0], data[:, 1], c=hidden_layer_output[:, 1], cmap='viridis', edgecolors='k')
    # 添加颜色条
    colorbar = plt.colorbar(scatter2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Hidden Layer Output visualization 2')
    plt.subplot(1, 3, 3)
    scatter3 = plt.scatter(data[:, 0], data[:, 1], c=hidden_layer_output[:, 2], cmap='viridis', edgecolors='k')
    # 添加颜色条
    colorbar = plt.colorbar(scatter3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Hidden Layer Output visualization 3')
    plt.show()


def visualize_output(model, data, targets):
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        x, y = data[:, 0], data[:, 1]
        x_min, x_max = x.min().item() - 0.5, x.max().item() + 0.5
        y_min, y_max = y.min().item() - 0.5, y.max().item() + 0.5
        points_x, points_y = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        points = torch.tensor(np.c_[points_x.flatten(), points_y.flatten()], dtype=torch.float32)
        points_output = model(points)
        points_z = (points_output > 0.5).float().reshape(points_x.shape)

    plt.contourf(points_x, points_y, points_z, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=targets.squeeze(), edgecolors='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Binary Classification for Concentric Circles')
    plt.show()
    # # 绘制3D图像
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(points_x, points_y, points_output.reshape(points_x.shape), cmap='viridis', rstride=1, cstride=1, alpha=0.5)
    # surface = ax.plot_surface(points_x, points_y, np.full(points_x.shape,0.5), cmap='viridis', rstride=1, cstride=1, alpha=0.5)
    # colorbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Model Output')
    # plt.show()
