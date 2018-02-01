import scripy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import random

def feedforward(w,a,x):
    ＃ sigmoid 激活函数
    f = lambda s: 1 / (1 + np.exp(-s))

    w = np.array(w)
    temp = np.array(np.concatenate(a,x),axis=0)
    z_next = np.dot(w , temp)
    return f(z_next), z_next


def backprop(w,z,delta_next):
    # sigmoid 激活函数
    f = lambda s: np.array(1 / (1 + np.exp(-s)))

    # sigmoid 激活函数的导数
    df = lambda s: f(s) * (1- f(s))

    delta = df(z) * np.dot(w.T, delta_next)
    return delta

DataSet = scio.loadmat('yaleB_face_dataset.mat')
unlabeledData = DataSet['unlabeled_data']

dataset_size = 80 #我们所准备无标签的人脸图片数据数量
unlabeled_data = np.zeros(unlabeledData.shape)

#利用z-score 归一化方法归一数据
for i in range(dataset_size):
    tmp = unlabeledData[:,i] / 255
    unlabeled_data[:,i] = (tmp - np.mean(tmp)) / np.std(tmp)

alpha = 0.5 ＃学习步长
max_epoch = 300 #自编码器训练总次数
mini_batch = 10 #最小批训练时，每次使用10个样本同时进行训练
height = 48 #人脸数据图片的高度
width = 42 #人脸数据图片的宽度
imgSize = height * width

#神经网络结构
hidden_node = 60 #网络隐藏层节点数目
hidden_layer = 2
layer_struc = [[imgSize, ,],[0, hidden_node],[0, imgSize]]
layer_num = 3 #网络层次数目

#初始化无监督网络的权值
w = []
for l in range(layer_num-1):
    w.append(np.random.randn(layer_struc[l+1][1],sum(layer_struc[1])))

#定义神经网络的外部节点数目
X = []
X.append(np.array(unlabeled_data[:,:]))
X.append(np.zeros((0,dataset_size)))
X.append(np.zeros((0,dataset_size)))

#初始化在网络训练过程中，进行误差反向传播所需的
delta = []
for l in range(layer_num):
    delta.append([])

# 定义结果展示参数
nRow = max_epoch / 100 +1
nColum = 4
eachFaceNum = 20  #对于每个人都有20张未标记图像数据

#在第一行中展示原始图像
for iImg in range(nColum):
    ax = plt.subplot(nRow, nColum, iImg+1)
    plt.imshow(unlabeledData[:,eachFaceNum * iImg + 1].reshape((width,height)).T, camp= plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#无监督训练
count = 0  #记录训练次数
print('Autoencoder training start..')
for iter in range(max_epoch):
    #定义随机洗牌下标
    ind = list(range(dataset_size))
    random.shuffle(ind)

    a = []
    z = []
    z.append([])
    for i in range(int(np.ceil(dataset_size / mini_batch))):
	a.append(np.zeros((layer_struc[0][1], mini_batch)))
	x = []
	for l in range(layer_num):
	    x.append( X[1][:,ind[i*mini_batch : min((i+1)*mini_batch, dataset_size)]])
	y = unlabeled_data[:,ind[i*mini_batch:min((i+1)*mini_batch,dataset_size)]]
	for l in range(layer_num-1):
	    a.append([])
	    z.append([])
	    a[l+1],z[l+1] = feedforward(w[l],a[l],x[l])

	delta[layer_num-1] = np.array(a[layer_num-1] - y) * np.array(a[layer_num -1])
	delta[layer_num-1] = delta[layer_num-1] * np.array(1-a[layer_num-1])
	for l in range(layer_num-2, 0, -1):
	    delta[1] = backprop(w[l],z[l],delta[l+1])

	for l in range(layer_num-1):
	    dw = np.dot(delta[l+1], np.concatenate((a[l],x[l]),axis=0).T) / mini_batch
	    w[l] = w[l] - alpha * dw
    count = count + 1

#每训练100次展示一次自编码器目前对原始图像的输出结果
    if np.mod(iter+1,100) == 0 :
	b = []
	b.append(np.zeros((layer_struc[0][1],dataset_size)))
	for l in range(layer_num-1):
	    tempA, tempZ = feedforward(w[l], b[l], X[l])
	    b.append(tempA)
	for iImg in range(nColum):
	    ax = plt.subplot(nEow,nColum, iImg + nColumn * (iter+1)/100 + 1)
	    tmp = b[layer_num-1][:,eachFaceNum * iImg + 1]
            dis_result = ((tmp * np.std(tmp)) + np.mean(tmp)).reshape(width,height).T
	    plt.imshow(dis_result,camp= plt.cm.gray)
	    ax.get_xaxis().set_visible(False)
    	    ax.get_yaxis().set_visible(False)
	print('Learning epoch:', count, '/', max_epoch)

fig2 = plt.figure(2)

#获得编码结果
code_result,tempZ = feedforward(w[0], b[0], X[0])

#展示原始数据图
for iImg in range(nColumn):
    ax = plt.subplot(2, nColumn, iImg+1)
    plt.imshow(unlabeled_data[:,eachFaceNum * iImg +1].reshape((width,height)).T, camp= plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#展示对应的编码结果
for iImg in range(nColumn):
    ax = plt.subplot(2,nColumn,iImg+nColumn+1)
    plt.imshow(code_result[:,eachFaceNum * iImg + 1].reshape((hidden_node,1)),camp=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


