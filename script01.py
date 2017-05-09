#!/bin/bash/env python
#-*- coding: utf-8
__author__ = 'ZhangJin'

'''
from tensorflow.examples.tutorials.mnist.input_data import *
import tensorflow as tf

mnist = read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None, 10])
keep_pro = tf.placeholder("float")
x_image = tf.reshape(x,[-1,28,28,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(a, w):
    return tf.nn.conv2d(a, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(a):
    return tf.nn.max_pool(a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_pro)


w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# # 执行循环
for i in range(2000):
    # 每批取出50个训练样本
    batch = mnist.train.next_batch(50)
    # 循环次数是100的倍数的时候，打印东东
    if i % 100 == 0:
        # 计算正确率，
        train_accuracy = accuracy.eval(feed_dict={
           x: batch[0], y_: batch[1], keep_pro: 1.0})
        # 打印
        print "step %d, training accuracy %g" % (i, train_accuracy)
    # 执行训练模型
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_pro: 0.5})
# 打印测试集正确率
print "test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_pro: 1.0
    })

'''
import numpy as np


class BPNN_zybb:
    # 初始化函数
    # 需要参数：1:输入样本numpy序列； 2：期望输出序列； 3：各个层的节点个数列表，由此列表可得到中间层的层数;
    def __init__(self, input_array, output_array, layer_node_count_list, w_step_size=0.02, b_step_size=0.02):
        self.layers = []  # 将要存储各层节点的值
        self.input_array = input_array
        # print(self.input_array)
        self.output_array = output_array  # 期望输出
        # print(self.output_array)
        self.w_step_size = w_step_size  # 控制逆向传播过程中权值系数调节的幅度
        self.b_step_size = b_step_size  # 控制逆向传播过程中阀值调节的幅度
        self.layer_node_count_list = layer_node_count_list
        # 根据 layers_list 可以得到神经网络层数
        self.layers_count = len(layer_node_count_list) - 1
        # 然后为每一层随机产生系数（numpy序列）,存储到字典对象之中
        self.w_dic = {}  # 权重
        self.b_dic = {}  # 阀值
        for i in range(1, len(layer_node_count_list)):
            self.w_dic[i - 1] = 2 * (
            np.random.random((layer_node_count_list[i - 1], layer_node_count_list[i])) - 0.5)  # 允许负值的存在
            self.b_dic[i - 1] = 2 * (np.random.random((1, layer_node_count_list[i])) - 0.5)  # 允许负值的存在
        # print (self.w_dic)
        # print(self.b_dic)
        # 如果数据归一化,存储两个最大特征值 与 最小特征值的array ,不归一化的话,不用考虑
        self.norm_max_array = None
        self.norm_min_array = None

    # 激活函数
    # f(x) = 1/(1+e**(-x))
    def sigmoid(self, x, derivative=False):  # derivative 为True ，代表求解激活函数的导数
        if (derivative == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # 正向传播
    def forward(self):
        self.layers = []  # 每次循环都要初始化self.layers 为空
        # 每一层以上一层的值为输入，值= 上层节点的值 * 系数
        self.layers.append(self.input_array)  # 首层节点的值已知
        for i in range(self.layers_count):
            laywer_value_raw = np.dot(self.layers[i], self.w_dic[i]) + self.b_dic[i]
            # 为该层每个节点代入激活函数求值
            laywer_value = self.sigmoid(laywer_value_raw)
            # 将该层节点的值，存入列表
            self.layers.append(laywer_value)
            # 这样子便得到了一次正向传播后所有的层的所有节点的值
            # print(self.layers)

    # 逆向传播
    # 正向传播很容易理解，但逆向传播过程就很抽象了
    # 很多时候仅仅分析此处的代码容易陷入迷茫：为什么这个变量a要与那个变量b相乘，而不是变量c，它们之间的逻辑关系在哪里？
    # 答案是：这里没有逻辑，一切的运算都是基于数学公式！
    # 逆向是bp神经网络的精髓，是以最小二乘法为原理，经过一系列数学推导而得出的公式，backword()函数就是公式的代码实现
    # 换句话说，没推倒过公式，就永远不了解bp神经网络为什么这么逆向运算，也就看不懂下面的backword()函数
    # 亲自推导一遍bp神经网络的数学公式后，你才会发现原来如此
    # 唠叨最后一句：公式很重要，一定要亲自去推导一遍
    def backword(self):
        # 逆向传播是将误差量按照系数由最后一层逐级返回到首层的过程
        # 误差量是期望输出与神经网络最后一层（输出层）的差值
        delta_list = []
        theta_output = (self.output_array - self.layers[-1]) * self.sigmoid(self.layers[-1],
                                                                            derivative=True)  # 输出层的反向传播 #（步长）
        delta_list.append(theta_output)
        # for循环 中间层的反向传播
        for i in range(self.layers_count - 1, 0, -1):
            theta_middle = np.dot(delta_list[-1], self.w_dic[i].T) * self.sigmoid(self.layers[i], derivative=True)
            delta_list.append(theta_middle)
        # 然后计算系数的调整量
        # print(len(delta_list))
        delta_list.reverse()
        w_change_dic = {}
        b_change_dic = {}
        for i in range(len(delta_list)):  # 与层数相等
            w_change_dic[i] = np.dot(self.layers[i].T, delta_list[i]) * self.w_step_size
            b_change_dic[i] = np.dot(np.array([[1]] * len(self.input_array)).T, delta_list[i]) * self.b_step_size
        # print(w_change_dic)
        # return w_change_dic
        # 在这里直接更新系数
        for i in w_change_dic.keys():
            self.w_dic[i] += w_change_dic[i]
            self.b_dic[i] += b_change_dic[i]

    # 余下的是一些随意添加的杂七杂八的模块 可以不看
    # 1 验证数据是否标准的模块(二维数据)
    def check_data(self):
        # 1 输入样本的长度应该与输出样本的长度相同
        input_shape = self.input_array.shape
        output_shape = self.output_array.shape
        # print(input_shape,output_shape)
        if len(input_shape) != 2:
            print("oo输入样本必须为2维的np序列，当前序列维度：{0}！".format(len(input_shape)))
            return

        if len(input_shape) != len(output_shape):  # 输入输出样本维度不同
            if len(output_shape) == 1:
                tmp_output_array = np.array([self.output_array]).T
                print(":&gt;将输出序列由{0}转化为:{1}".format(self.output_array, tmp_output_array))
                self.output_array = tmp_output_array
                output_shape = self.output_array.shape

        if input_shape[0] == output_shape[0]:
            # print("-v-输入输出样本长度相同，正确.")
            pass
        else:
            if output_shape[0] == 1 and input_shape[0] == output_shape[1]:  #
                print(":&gt;将输出序列由{0}转化为:{1}".format(self.output_array, self.output_array.T))
                self.output_array = self.output_array.T
                output_shape = self.output_array.shape
            else:
                print("oo输入输出样本的长度不同，无法继续处理！")
                return
        # 至此说明样本没问题，然后检查输入是否出错
        if input_shape[1] != self.layer_node_count_list[0]:
            print("oo输入层节点数目是错误的，输入为：{0}，而根据数据（输入特征数目）应该为{1}!".format(self.layer_node_count_list[0], input_shape[1]))
            return
        if output_shape[1] != self.layer_node_count_list[-1]:
            print("oo输出层节点数目是错误的，输入为：{0}，而根据数据（输出特征数目）应该为{1}!".format(self.layer_node_count_list[-1], output_shape[1]))
            return
        print("-v- 数据正确，可以开始运算了……")

    # 预测函数
    # 对于已经执行了bp神经网络的对象,调用次函数,输入array数据,即可根据训练得到的w与b得到预测输出(其实只需要调用forward函数即可)
    # pre_input_array 为输入数据 , normalization为True代表是否需要归一化,因为如果训练时用到归一化,那么预测时也需要调用归一化函数
    def preview(self, pre_input_array):
        # 检查输入数据
        input_data_shape = pre_input_array.shape  # 输入数据的形状
        if len(input_data_shape) != 2:
            if len(input_data_shape) == 1:
                print("预测:输入数据维度为1,尝试修正...")
                pre_input_array = np.array([pre_input_array])
            else:
                print("预测:输入数据形状为{0},无法修正,退出!".format(input_data_shape))
                exit(0)
        # 下面开始预测,forward函数的细微修改
        pre_layers = []  # 每次循环都要初始化self.layers 为空
        # 每一层以上一层的值为输入，值= 上层节点的值 * 系数
        pre_layers.append(pre_input_array)  # 首层节点的值已知
        for i in range(self.layers_count):
            laywer_value_raw = np.dot(pre_layers[i], self.w_dic[i]) + self.b_dic[i]
            # 为该层每个节点代入激活函数求值
            laywer_value = self.sigmoid(laywer_value_raw)
            # 将该层节点的值，存入列表
            pre_layers.append(laywer_value)
        print("预测输出值为:{0}".format(pre_layers[-1].T))
        return pre_layers




if __name__ == '__main__':
    from sklearn.datasets import make_moons

    x, y = make_moons(250, noise=0.25)

    x_train = x[:200]
    y_train = y[:200]

    x_pre = x[200:]
    y_pre = y[200:]

    print("开始计算……")
    bpnn = BPNN_zybb(x_train, y_train, [2, 3, 4, 1], w_step_size=0.05)
    bpnn.check_data()
    for i in range(1000):
        print(i)
        bpnn.forward()
        bpnn.backword()
    # print(bpnn.layers[-1].T)
    # print(y.T)
    print("训练结束,预测开始...")
    layers = bpnn.preview(x_pre)
    # 预测与输期望出值的对比
    pre_result = []
    for i in range(len(layers[-1])):
        a_data = [layers[-1][i][0], y_pre[i]]
        print(a_data)
        pre_result.append(a_data)
    print("\n")
    print(pre_result)

if __name__ == '__main__':
    from sklearn.datasets import make_moons

    x, y = make_moons(250, noise=0.25)

    x_train = x[:200]
    y_train = y[:200]

    x_pre = x[200:]
    y_pre = y[200:]

    print("开始计算……")
    bpnn = BPNN_zybb(x_train, y_train, [2, 3, 4, 1], w_step_size=0.05)
    bpnn.check_data()
    for i in range(1000):
        print(i)
        bpnn.forward()
        bpnn.backword()
    # print(bpnn.layers[-1].T)
    # print(y.T)
    print("训练结束,预测开始...")
    layers = bpnn.preview(x_pre)
    # 预测与输期望出值的对比
    pre_result = []
    for i in range(len(layers[-1])):
        a_data = [layers[-1][i][0], y_pre[i]]
        print(a_data)
        pre_result.append(a_data)
    print("\n")
    print(pre_result)

'''

a = np.array([[1, 1],[2,3]])
b = np.array([[4, 5],[6,7]])

print (a-b)
print np.sum(a,0)/float(2)
'''