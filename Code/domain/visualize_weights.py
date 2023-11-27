# -- coding: utf-8 --
# @Time : 2023/11/9 19:39
# @Author : Tsubaki_01
# @File : visualize_weights.py
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def visualize_weights_and_bias_through_tensorboard(model, epoch, logdir='./logs'):
    model.eval()
    with torch.no_grad():
        writer = SummaryWriter(logdir)

        fc1_weights = model.fc1.weight.tolist()
        fc2_weights = model.fc2.weight.tolist()

        for i in range(len(fc1_weights)):
            fc1_weight = fc1_weights[i]
            writer.add_scalars(f'the {i + 1}th neuro weights of fc1 layer',
                               {f'weight{j}': fc1_weight[j] for j in range(len(fc1_weight))}, epoch)
        for i in range(len(fc2_weights)):
            fc2_weight = fc2_weights[i]
            writer.add_scalars(f'the {i + 1}th neuro weights of fc2 layer',
                               {f'weight{j + 1}': fc2_weight[j] for j in range(len(fc2_weight))}, epoch)

        writer.close()
        # for layer, weights in model.named_parameters():
        #     if layer
        #         weights = weights.tolist()
        #         for i in range(len(weights)):
        #             writer.add_scalars(f'the {i}th neuro weights of layer {layer}',
        #                                weights[i], epoch)
        #             writer.add_scalar()
        # print(1)
