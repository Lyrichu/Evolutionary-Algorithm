# -*- coding:utf-8 -*-
"""
Visualize Genetic Algorithm to find a maximum point in a function.

Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10            # DNA length
POP_SIZE = 100           # population size(种群大小)
CROSS_RATE = 0.8         # mating probability (DNA crossover)交叉概率
MUTATION_RATE = 0.003    # mutation probability变异概率
N_GENERATIONS = 200      # 进化代数
X_BOUND = [0, 5]         # x upper and lower bounds(变量取值)


def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function(寻找函数最大值)


# find non-zero fitness for selection
# 获取非零的适应度
def get_fitness(pred): return pred + 1e-3 - np.min(pred)


# convert binary DNA to decimal and normalize it to a range(0, 5)
# 将二进制的DNA转化为实数，并且取值变为(0,5)
# pop 为长为 DNA_SIZE的二进制串
def translateDNA(pop): return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]

# 选择操作
# 采用轮盘赌策略选择
def select(pop, fitness):    # nature selection wrt pop's fitness
	# 以轮盘赌策略从种群中选择某一个个体,idx 为个体在种群序号
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
	# 返回选择的个体
    return pop[idx]

# 交叉操作
def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)
		# select another individual from pop
		# 长为 DNA_SIZE 的bool向量
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent

# 变异操作
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

# 初始化操作
pop = np.random.randint(0, 2, (1, DNA_SIZE)).repeat(POP_SIZE, axis=0)  # initialize the pop DNA

# 开启交互式操作
plt.ion()       # something about plotting
# 0到5产生两百个数，为等差数列
x = np.linspace(*X_BOUND, 200)
# 绘制函数图像
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
	# 计算种群每个个体的适应度值
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA

    # something about plotting
    if 'sca' in globals(): sca.remove()
	# 绘制当前代数种群中的所有个体点
    sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # GA part (evolution)
	# 计算适应度值
    fitness = get_fitness(F_values)
	# 打印适应度最高的个体(DNA形式)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
	# 选择操作
    pop = select(pop, fitness)
	# 复制个体
    pop_copy = pop.copy()
    for parent in pop:
		# 交叉操作
        child = crossover(parent, pop_copy)
		# 变异操作
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child

plt.ioff(); plt.show()