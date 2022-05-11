#! -*- coding: utf-8 -*-
import pickle
import numpy as np
from config import maxlen
from loader import tokenizer, load_data
from model import get_model
from path import test_file_path

mlb = pickle.load(open('./weights/mlb.pkl', 'rb'))
print(mlb.classes_.tolist())
threshold = 0.5
model_weights = "./weights/multi-label_roformer_v2_AdamEMA.h5"
model = get_model()
model.load_weights(model_weights)


def predict(test_text):
    token_ids, segment_ids = tokenizer.encode(test_text, maxlen = maxlen)
    pred = model.predict([[token_ids], [segment_ids]])
    
    label_index = np.where(pred[0] > threshold)[0]  # 取概率值大于阈值的 onehot 向量索引, [12,34]
    labels = [mlb.classes_.tolist()[i] for i in label_index]
    one_hot_label = np.where(pred[0] > threshold, 1, 0)  # [[0,0,1,0,0,..],[0,0,1,0,1,..]]
    return one_hot_label, '|'.join(labels)


if __name__ == '__main__':
    test_text = '美国芝加哥北部郊区附近发生爆炸至少4人受伤'
    one_hot, label = predict(test_text)
    print("测试样本预测标签：", label)