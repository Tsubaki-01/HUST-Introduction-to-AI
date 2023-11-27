import math
import matplotlib.pyplot as plt
import numpy as np

DNA_SIZE = 18  # DNA长度
POP_SIZE = 100  # 种群大小
CROSSOVER_RATE = 0.5  # 交叉概率
MUTATION_RATE = 0.01  # 变异概率
ITERATIONS = 50  # 迭代代数
X_DOD = [0, 10]  # X定义域


def Y(x):  # 求y
    return 10 * np.sin(5 * x) + 7 * abs(x - 5) + 10


def decode_DNA(pop):  # 解码DNA
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) * (X_DOD[1] - X_DOD[0]) / float(2 ** (DNA_SIZE - 1)) + X_DOD[0]


def get_fitness(pop):  # 适应度函数
    x = decode_DNA(pop)
    y = Y(x)
    return -(y - np.max(y)) + 0.001


def select(pop, fitness):  # 选择
    selected = np.random.choice(np.arange(POP_SIZE), POP_SIZE, replace=True, p=fitness / (fitness.sum()))
    return pop[selected]


def crossover(pop, crossover_rate):  # 交叉
    new_pop = []
    for individual in pop:
        temp = individual
        if np.random.rand() < crossover_rate:
            another_individual = pop[np.random.randint(POP_SIZE)]
            cross_point1 = np.random.randint(0, DNA_SIZE - 1)
            cross_point2 = np.random.randint(cross_point1, DNA_SIZE)
            temp[cross_point1: cross_point2] = another_individual[cross_point1: cross_point2]
        mutation(temp, MUTATION_RATE)
        new_pop.append(temp)
    return new_pop


def mutation(mutate_one, mutation_rate):  # 变异
    if np.random.rand() < mutation_rate:
        mutated_point = np.random.randint(0, DNA_SIZE)
        mutate_one[mutated_point] = mutate_one[mutated_point] ^ 1


def print_info_of_pop(pop):  # 输出代际最优
    fitness = get_fitness(pop)
    index = np.argmax(fitness)
    print("the biggest fitness: ", fitness[index])
    x = decode_DNA(pop)
    print("DNA encode: ", pop[index])
    print("DNA decode: ", x[index])
    print("y_min: ", Y(x[index]))


def plot_2d(x, y):  # 获取折线图画布
    fig = plt.figure()

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(X_DOD)
    plt.xticks(np.arange(*X_DOD), labels=[str(i) for i in np.arange(*X_DOD)])

    return fig


if __name__ == '__main__':
    x_line = np.linspace(*X_DOD, 1000)
    y_line = Y(x_line)
    fig = plot_2d(x_line, y_line)  # 画函数图
    # 迭代
    generation = np.random.randint(0, 2, (POP_SIZE, DNA_SIZE))
    for ite in range(ITERATIONS):
        # 画散点图
        x_scatter = decode_DNA(generation)
        y_scatter = Y(x_scatter)
        if 'scat' in locals():
            scat.remove()
        scat = plt.scatter(x_scatter, y_scatter, c='red', marker='+')

        plt.pause(0.1)
        # 交叉选择
        generation = np.array(crossover(generation, CROSSOVER_RATE))
        generation = select(generation, get_fitness(generation))

        if np.max(get_fitness(generation)) - np.min(get_fitness(generation)) < 0.1:
            break

    print_info_of_pop(generation)
    print(f'used {ite} iterations')