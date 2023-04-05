import tensorflow as tf
import numpy as np

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # train_data.shape = (60000, 28, 28): 60000個28x28的input
        # 將像素都坐正規化，並且增加一個維度存放channel數量(1,因為為灰階)
        # 用astype()轉型
        self.train_data = np.expand_dims(self.train_data.astype(np.float32)/255.0, axis=-1) #[60000,28,28,1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32)/255.0, axis=-1) #[10000,28,28,1]
        self.train_label = self.train_label.astype(np.int32) #[60000]
        self.test_label = self.test_label.astype(np.int32) #[10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
        
    def get_batch(self, batch_size):
        # 從資料集中隨機存取batch_size個元素return
        # index為存放的要挑選的index(一維陣列)
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]
        
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten層將除了第一維(batch_size)以外的維度攤平
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
    
    def call(self, inputs):      # [batch_size, 28, 28, 1]
        x = self.flatten(inputs) # [batch_size, 784]
        x = self.dense1(x)       # [batch_size, 100]
        x = self.dense2(x)       # [batch_size, 8]
        output = tf.nn.softmax(x)
        return output