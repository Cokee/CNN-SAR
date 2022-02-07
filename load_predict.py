import os.path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

tf.compat.v1.disable_eager_execution()  # 清理开始的会话或者图的代码。使其为空。


def load_ckpt_model(sess, save_path):
    checkpoint = tf.train.get_checkpoint_state(save_path)  # 从checkpoint文件中读取checkpoint对象
    input_checkpoint = checkpoint.model_checkpoint_path
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)  # 加载模型结构
    saver.restore(sess, input_checkpoint)  # 使用最新模型
    sess.run(tf.global_variables_initializer()) # 初始化所有变量


# 加载计算图结构
sess = tf.Session()
load_ckpt_model(sess, './model/')
# 获取计算图节点
graph = tf.get_default_graph()  # 获取计算图
tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
variable_names = [v.name for v in tf.trainable_variables()]
input_x = graph.get_tensor_by_name("input_x:0")

outputs = graph.get_tensor_by_name("outputs:0")