import numpy as np
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence

file_path = r'D:\Phyon\Project\bys\label_data_change1.xlsx'  # 读取表格
data = pd.read_excel(file_path,header=0)  # header为表头，自动去掉表头
data = data.values

N = 8
features = data[:, 1:N+1]  # 特征
labels_q = data[:, -3]     # 是否合格
labels_qd = data[:, -4]     # 偏差
labels_en = data[:, -2]     # 能耗


# 划分训练集，4:1
features_train, features_test,labels_q_train, labels_q_test,labels_qd_train, labels_qd_test,labels_en_train,\
labels_en_test = train_test_split(features, labels_q,labels_qd,labels_en,test_size=0.2, random_state=0)


'''
随机数种子是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，
其他参数一样的情况下你得到的随机数组是一样的。不填的话默认值为False，即每次切分的比例虽然相同，但是切分的结果不同
'''


# # 标准化
# from sklearn.preprocessing import StandardScaler
# ss_x,ss_y = StandardScaler(),StandardScaler()
# features_train = ss_x.fit_transform(features_train)
# features_test = ss_x.transform(features_test)
# labels_q_train = ss_y.fit_transform(labels_q_train.reshape([-1,1])).reshape(-1)
# labels_q_test = ss_y.transform(labels_q_test.reshape([-1,1])).reshape(-1)
# labels_qd_train = ss_y.fit_transform(labels_qd_train.reshape([-1,1])).reshape(-1)
# labels_qd_test = ss_y.transform(labels_qd_test.reshape([-1,1])).reshape(-1)
# labels_en_train = ss_y.fit_transform(labels_en_train.reshape([-1,1])).reshape(-1)
# labels_en_test = ss_y.transform(labels_en_test.reshape([-1,1])).reshape(-1)

from sklearn.metrics import mean_squared_error
def plot_learning_curve(reg, X_train, X_test, y_train, y_test):
    # 使用线性回归绘制学习曲线
    train_score = []
    test_score = []

    for i in range(1, len(X_train)):
        reg.fit(X_train[:i], y_train[:i])
        y_train_predict = reg.predict(X_train[:i])
        y_test_predict = reg.predict(X_test)
        train_score.append(mean_squared_error(y_train_predict, y_train[:i]))
        test_score.append(mean_squared_error(y_test_predict, y_test))

    plt.plot([i for i in range(1, len(X_train))], np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train))], np.sqrt(test_score), label="test")

    plt.legend()
    plt.show()



# 随机森林
print('分类合格品')
print('开始训练随机森林 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
q_rf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
q_rf.fit(features_train, labels_q_train)
print('随机森林训练完毕 | ','训练集分数为',q_rf.score(features_train, labels_q_train),"验证集分数为", q_rf.score(features_test, labels_q_test))   # 决定系数R^2
print("--" * 100)

labels_q_predict = q_rf.predict(features_test)
labels_q_train_predict = q_rf.predict(features_train)

from sklearn.metrics import f1_score
f1 = f1_score (labels_q_test, labels_q_predict, labels=None, pos_label=1, average='binary', sample_weight=None)
f1_1 = f1_score (labels_q_train, labels_q_train_predict, labels=None, pos_label=1, average='binary', sample_weight=None)
feat_labels1 = ["余量","切换位置","周期","1速","2速","保压压力","最大压力","最大速度"]
feat_labels=pd.DataFrame(feat_labels1)
importances = q_rf.feature_importances_
importances_q = importances
indices = np.argsort(importances)[::-1] # 下标排序
print('特征排序：')
for f in range(8):
    print("%2d) %-*s %f" % \
          (f + 1, 30, feat_labels1[indices[f]], importances[indices[f]]))
print("--" * 100)



# 训练随机森林,模型输出为偏差
print('预测偏差')
from sklearn.ensemble import RandomForestRegressor
print('开始训练随机森林 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
qd_rf = RandomForestRegressor(n_estimators=51,max_depth=10,min_samples_split=3,criterion='absolute_error')
qd_rf.fit(features_train, labels_qd_train)
print('随机森林训练完毕 | ','训练集分数为',qd_rf.score(features_train, labels_qd_train),"验证集分数为", qd_rf.score(features_test, labels_qd_test))   # 决定系数R^2
print("--" * 100)

labels_qd_predict = qd_rf.predict(features_test)
r_qd = labels_qd_test - labels_qd_predict

labels_qd_train_predict = qd_rf.predict(features_train)

importances = qd_rf.feature_importances_
importances_qd = importances
indices = np.argsort(importances)[::-1] # 下标排序
print('特征排序：')
for f in range(8):
    print("%2d) %-*s %f" % \
          (f + 1, 30, feat_labels1[indices[f]], importances[indices[f]]))
print("--" * 100)


# 训练随机森林,模型输出为能耗
print('预测能耗')
print('开始训练随机森林 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
en_rf = RandomForestRegressor()
en_rf.fit(features_train, labels_en_train)
print('随机森林训练完毕 | ','训练集分数为',en_rf.score(features_train, labels_en_train),"验证集分数为", en_rf.score(features_test, labels_en_test))   # 决定系数R^2
print("--" * 100)

labels_en_predict = en_rf.predict(features_test)
r_en = labels_en_test - labels_en_predict
labels_en_train_predict = en_rf.predict(features_train)

importances = en_rf.feature_importances_
importances_en = importances
indices = np.argsort(importances)[::-1] # 下标排序
print('特征排序：')
for f in range(8):
    print("%2d) %-*s %f" % \
          (f + 1, 30, feat_labels1[indices[f]], importances[indices[f]]))
print("--" * 100)



features1 = np.array(features)
# 遗传算法优化
# 参数
DNA_SIZE = 12   # DNA长度与保留位数有关,长度越长精度越高,10的倍数
POP_ORI_SIZE = 500000  # 初始种群数量
CROSSOVER_RATE = 0.8    # 交叉概率
MUTATION_RATE = 0.01   # 变异概率
N_GENERATIONS = 100  # 迭代次数
X1_BOUND = [min(features1[:,0]), max(features1[:,0])]   # 范围
X2_BOUND = [min(features1[:,1]), max(features1[:,1])]
X3_BOUND = [min(features1[:,2]), max(features1[:,2])]
X4_BOUND = [min(features1[:,3]), max(features1[:,3])]
X5_BOUND = [min(features1[:,4]), max(features1[:,4])]
X6_BOUND = [min(features1[:,5]), max(features1[:,5])]   # 范围
X7_BOUND = [min(features1[:,6]), max(features1[:,6])]
X8_BOUND = [min(features1[:,7]), max(features1[:,7])]



# 求最小值适应度函数
def get_fitness(pop):
    x1,x2,x3,x4,x5,x6,x7,x8 = translateDNA(pop)
    a = [x1,x2,x3,x4,x5,x6,x7,x8]
    c = np.array(a)
    x = c.transpose()
    # pred1 = qd_rf.predict(x)
    pred11 = qd_rf.predict(x)
    pred22 = en_rf.predict(x)

    aa = 0
    bb = 1

    cc1 = min(pred11)
    dd1 = max(pred11)
    k1 = (bb-aa)/(dd1-cc1)
    pred1 = aa + k1 * (pred11 - cc1)

    cc2 = min(pred22)
    dd2 = max(pred22)
    k2 = (bb-aa)/(dd2-cc2)
    pred2 = aa + k2 * (pred22 - cc2)

    pred = 0.1*pred1 + 0.9*pred2
    return -(pred - np.max(pred)) + 1e-3


# 解码过程
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x1_pop = pop[:, ::8]   # 从第一个元素起，步长为8取元素
    x2_pop = pop[:, 1::8]  # 从第二个元素起，步长为8取元素
    x3_pop = pop[:, 2::8]  # 从第三个元素起，步长为8取元素
    x4_pop = pop[:, 3::8]  # 从第四个元素起，步长为8取元素
    x5_pop = pop[:, 4::8]  # 从第五个元素起，步长为8取元素
    x6_pop = pop[:, 5::8]   # 从第六个元素起，步长为8取元素
    x7_pop = pop[:, 6::8]  # 从第七个元素起，步长为8取元素
    x8_pop = pop[:, 7::8]  # 从第八个元素起，步长为8取元素

    x1 = x1_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X1_BOUND[1] - X1_BOUND[0]) + X1_BOUND[0]
    x2 = x2_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X2_BOUND[1] - X2_BOUND[0]) + X2_BOUND[0]
    x3 = x3_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X3_BOUND[1] - X3_BOUND[0]) + X3_BOUND[0]
    x4 = x4_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X4_BOUND[1] - X4_BOUND[0]) + X4_BOUND[0]
    x5 = x5_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X5_BOUND[1] - X5_BOUND[0]) + X5_BOUND[0]
    x6 = x6_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X6_BOUND[1] - X6_BOUND[0]) + X6_BOUND[0]
    x7 = x7_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X7_BOUND[1] - X7_BOUND[0]) + X7_BOUND[0]
    x8 = x8_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X8_BOUND[1] - X8_BOUND[0]) + X8_BOUND[0]

    return x1,x2,x3,x4,x5,x6,x7,x8


# 交叉过程（过程中产生变异）
def crossover_and_mutation(pop, CROSSOVER_RATE):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child,MUTATION_RATE)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


# 变异过程
def mutation(child, MUTATION_RATE):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


# 选择过程
def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


# 打印结果
def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    # print("max_fitness:", fitness[max_fitness_index])
    x1,x2,x3,x4,x5,x6,x7,x8 = translateDNA(pop)
    print('\n')
    print('遗传算法优化完毕 | ', '时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("--" * 100)
    print("最优参数为:", (x1[max_fitness_index], x2[max_fitness_index],x3[max_fitness_index],x4[max_fitness_index],x5[max_fitness_index],x6[max_fitness_index], x7[max_fitness_index],x8[max_fitness_index]))
    print("最低偏差为:",qd_rf.predict([[x1[max_fitness_index], x2[max_fitness_index],x3[max_fitness_index],x4[max_fitness_index],x5[max_fitness_index],x6[max_fitness_index], x7[max_fitness_index],x8[max_fitness_index]]])[0])
    print("最低能耗为:",en_rf.predict([[x1[max_fitness_index], x2[max_fitness_index], x3[max_fitness_index], x4[max_fitness_index],x5[max_fitness_index],x6[max_fitness_index], x7[max_fitness_index],x8[max_fitness_index]]])[0])

en_obj = []
qd_obj = []

def obj(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    x1, x2, x3, x4, x5, x6, x7, x8 = translateDNA(pop)
    qd_obj.append(qd_rf.predict([[x1[max_fitness_index], x2[max_fitness_index], x3[max_fitness_index],
                                    x4[max_fitness_index], x5[max_fitness_index], x6[max_fitness_index],
                                    x7[max_fitness_index], x8[max_fitness_index]]])[0])
    en_obj.append(en_rf.predict([[x1[max_fitness_index], x2[max_fitness_index], x3[max_fitness_index],
                                    x4[max_fitness_index], x5[max_fitness_index], x6[max_fitness_index],
                                    x7[max_fitness_index], x8[max_fitness_index]]])[0])


xx,yy=np.shape(features)
features1 = np.zeros([xx,yy])  # 存放标准化到[0,4095]的数

def demoo(value):
    a=0
    b=4095   # 12位二进制数
    k = (b-a)/(max(value)-min(value))
    return [a+k*(x-min(value)) for x in value]


for j in range(yy):
    features1[:,j] = demoo(features[:,j])

features2 = np.round(features1)  # 四舍五入

x11 = features2[:,0]
x22 = features2[:,1]
x33 = features2[:,2]
x44 = features2[:,3]
x55 = features2[:,4]
x66 = features2[:,5]
x77 = features2[:,6]
x88 = features2[:,7]

x1 = [bin(int(x11[i]))[2:] for i in range(xx)]
x2 = [bin(int(x22[i]))[2:] for i in range(xx)]
x3 = [bin(int(x33[i]))[2:] for i in range(xx)]
x4 = [bin(int(x44[i]))[2:] for i in range(xx)]
x5 = [bin(int(x55[i]))[2:] for i in range(xx)]
x6 = [bin(int(x66[i]))[2:] for i in range(xx)]
x7 = [bin(int(x77[i]))[2:] for i in range(xx)]
x8 = [bin(int(x88[i]))[2:] for i in range(xx)]

xxx = np.array([x1,x2,x3,x4,x5,x6,x7,x8])
xxxx = xxx.transpose()

for j in range(yy):
    for i in range(xx):
        l = len(xxxx[i,j])
        pr = 12-l
        pr0 = pr*'0'
        xxxx[i,j]= '{}{}'.format(pr0,xxxx[i,j])


x_initial = np.zeros([xx,12*N])

jj = 0
for i in range(0,12*N,12):
    for j in range(xx):
        x_initial[j, i] = int(xxxx[j,jj][0])
        x_initial[j, i + 1] = int(xxxx[j, jj][1])
        x_initial[j, i + 2] = int(xxxx[j, jj][2])
        x_initial[j, i + 3] = int(xxxx[j, jj][3])
        x_initial[j, i + 4] = int(xxxx[j, jj][4])
        x_initial[j, i + 5] = int(xxxx[j, jj][5])
        x_initial[j, i + 6] = int(xxxx[j, jj][6])
        x_initial[j, i + 7] = int(xxxx[j, jj][7])
        x_initial[j, i + 8] = int(xxxx[j, jj][8])
        x_initial[j, i + 9] = int(xxxx[j, jj][9])
        x_initial[j, i + 10] = int(xxxx[j, jj][10])
        x_initial[j, i + 11] = int(xxxx[j, jj][11])
    jj = jj+1

x_initial1 = np.zeros([xx,12*N])

x_initial1[:, 0::8] = x_initial[:,0:12]
x_initial1[:, 1::8] = x_initial[:, 12:24]
x_initial1[:, 2::8] = x_initial[:, 24:36]
x_initial1[:, 3::8] = x_initial[:, 36:48]
x_initial1[:, 4::8] = x_initial[:, 48:60]
x_initial1[:, 5::8] = x_initial[:, 60:72]
x_initial1[:, 6::8] = x_initial[:, 72:84]
x_initial1[:, 7::8] = x_initial[:, 84:96]

x_initial2 = x_initial1.astype(int)

# 优化
print('开始进行遗传算法优化 | ','时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),'\n')
# pop = np.random.randint(2, size=(POP_ORI_SIZE, DNA_SIZE * N))   # 随机生成基因型
pop = x_initial2
POP_SIZE = len(x_initial1)
for i in range(N_GENERATIONS):  # 迭代N代

    if POP_SIZE != POP_ORI_SIZE:
        xx,yy = pop.shape
        pop_add = np.random.randint(2, size=(POP_ORI_SIZE-POP_SIZE, DNA_SIZE * N))
        pop = np.vstack((pop, pop_add))
        POP_SIZE = POP_ORI_SIZE
    x1, x2, x3, x4, x5, x6, x7, x8 = translateDNA(pop)
    fitness = get_fitness(pop)
    max_fitness_index1 = np.argmax(fitness)
    max_fitness1=fitness[max_fitness_index1]
    pop = select(pop, fitness)  # 选择生成新的种群
    pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))  # 进行选择

    x1, x2, x3, x4, x5, x6, x7, x8 = translateDNA(pop)
    POP_SIZE1 = POP_SIZE
    # 根据质量是否合格剔除个体
    a = [x1,x2,x3,x4,x5,x6,x7,x8]  #
    de1 = [index for index in range(POP_ORI_SIZE) if (x4[index]-x5[index])<5]
    c = np.array(a)
    x = c.transpose()
    pp = q_rf.predict(x)  # 预测质量是否合格
    # print(pp)
    de2 = [index for (index, value) in enumerate(pp) if value == 0]  # 记录不合格的个体的索引值
    de = list(set(de1+de2))
    pop = np.delete(pop, de, axis=0)  # 剔除个体
    POP_SIZE = POP_SIZE1 - len(de)
    obj(pop)

    # if (i+1)%10==0:
    # print('已完成'+str(int((i+1)*100/N_GENERATIONS))+'% | ', '时间：', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # print('已迭代'+str(i+1)+'代 | ','共剔除'+str(POP_ORI_SIZE-POP_SIZE)+'个样本，'+'当前种群数量为：', str(POP_SIZE), '\n')

    print('已迭代'+str(i+1)+'代|'+'共'+str(N_GENERATIONS)+'代'+'|当前最优的适应度函数值为'+str(max_fitness1))

print_info(pop)
# plt.plot([i for i in range(1, len(en_obj)+1)], en_obj, label="能耗")
# plt.legend()
# plt.show()
#
# plt.plot([i for i in range(1, len(qd_obj)+1)], qd_obj, label="偏差")
# plt.legend()
# plt.show()
#
