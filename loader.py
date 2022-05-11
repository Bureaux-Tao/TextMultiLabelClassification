import json
import random
import pandas as pd
import numpy as np

from config import dict_path, maxlen
from utils.snippets import DataGenerator, sequence_padding
from utils.tokenizers import Tokenizer


def load_data(file_path):
    """加载数据
    单条格式：(文本, 标签id)
    """
    sample_list = []
    label_list = []
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in f.readlines():
            line = line.strip().split()  # line = ['label1|label2','sample_text']
            label_list.append(line[0].split('|'))
            sample_list.append(line[1])
    
    text_len = [len(text) for text in sample_list]
    df = pd.DataFrame()
    df['len'] = text_len
    print('训练文本长度分度')
    print(df['len'].describe())
    
    return sample_list, label_list


tokenizer = Tokenizer(dict_path)


class data_generator(DataGenerator):
    """
    数据生成器
    """
    
    def __iter__(self, random = False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen = maxlen)  # [1,3,2,5,9,12,243,0,0,0]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], np.asarray(batch_labels)
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


if __name__ == '__main__':
    x, y = load_data('./data/multi-classification-train.txt')
    y_list = []
    for i in y:
        y_list.append(i[0])
    print(len(list(set(y_list))))
