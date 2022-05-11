#! -*- coding: utf-8 -*-
import json
import pickle
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

from config import maxlen, batch_size
from model import model
from path import weights_path, event_type, MODEL_TYPE, train_file_path, test_file_path
from utils.adversarial import adversarial_training
from utils.backend import keras
from utils.loss import binary_focal_loss
from utils.optimizers import Adam, extend_with_exponential_moving_average
from loader import load_data, tokenizer, data_generator


def cal_acc(text_list, label_label):
    cnt = 1e-10
    total = len(text_list)
    for text, label in tqdm(zip(text_list, label_label)):
        token_ids, segment_ids = tokenizer.encode(text, maxlen = maxlen)
        pred = model.predict([[token_ids], [segment_ids]])
        pred = np.where(pred[0] > 0.5, 1, 0)
        cnt += 1 - (label != pred).any()
    
    return cnt / total


optimizer_name = "AdamEMA"
AdamEMA = extend_with_exponential_moving_average(Adam, name = optimizer_name)
optimizer = AdamEMA(lr = 5e-5)
model.compile(
    loss = [binary_focal_loss(alpha = .25, gamma = 2)],  # 二分类交叉熵损失函数
    optimizer = optimizer,
    metrics = ['accuracy'],
)

adversarial_training(model, 'Embedding-Token', 0.5)


class Evaluator(keras.callbacks.Callback):
    def __init__(self, patience = 5):
        super().__init__()
        self.patience = patience
        self.best_acc = 0
    
    def on_train_begin(self, logs = None):
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch, logs = None):
        # model.load_weights(bast_model_filepath)
        optimizer.apply_ema_weights()
        acc = cal_acc(test_x, test_y)  # 计算多标签分类准确率
        if acc > self.best_acc:
            self.best_acc = acc
            self.wait = 0
            save_best_path = "{}/{}_{}_{}.h5".format(weights_path, event_type, MODEL_TYPE, optimizer_name)
            model.save_weights(save_best_path)
        else:
            self.wait += 1
            print("Early stop count " + str(self.wait) + "/" + str(self.patience))
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        optimizer.reset_old_weights()
        print('acc: %.4f best acc: %.4f\n' % (acc, self.best_acc))
    
    def on_train_end(self, logs = None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


train_x, train_y = load_data(train_file_path)
test_x, test_y = load_data(test_file_path)

shuffle_index = [i for i in range(len(train_x))]
random.shuffle(shuffle_index)
train_x = [train_x[i] for i in shuffle_index]
train_y = [train_y[i] for i in shuffle_index]

mlb = MultiLabelBinarizer()
mlb.fit(train_y)
print("标签数量：", len(mlb.classes_))
class_nums = len(mlb.classes_)
pickle.dump(mlb, open(weights_path + '/mlb.pkl', 'wb'))

train_y = mlb.transform(train_y)  # [[label1,label2],[label3]] --> [[1,1,0],[0,0,1]]
test_y = mlb.transform(test_y)

train_data = [[x, y.tolist()] for x, y in zip(train_x, train_y)]  # 将相应的样本和标签组成一个tuple
print(train_data[:3])
test_data = [[x, y.tolist()] for x, y in zip(test_x, test_y)]  # --> [[x1,y1],[x2,y2],[],..]

# 转换数据集
train_generator = data_generator(train_data, batch_size)
test_generator = data_generator(test_data, batch_size)

evalutor = Evaluator(5)

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch = len(train_generator),
    epochs = 999,
    shuffle = True,
    callbacks = [evalutor]
)
