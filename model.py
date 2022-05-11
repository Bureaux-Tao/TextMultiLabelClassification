from keras.utils import plot_model

from config import config_path, checkpoint_path, class_nums
from path import MODEL_TYPE
from utils.backend import keras, K, set_gelu
from utils.models import build_transformer_model
from utils.optimizers import extend_with_exponential_moving_average, Adam

set_gelu('tanh')  # relu


class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """
    
    def __init__(self, layer, lamb, is_ada = False):
        self.layer = layer
        self.lamb = lamb  # 学习率比例
        self.is_ada = is_ada  # 是否自适应学习率优化器
    
    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma',
                    'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb  # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb ** 0.5  # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb)  # 更改初始化
                setattr(self.layer, key, weight * lamb)  # 按比例替换
        return self.layer(inputs)


def get_model():
    # 加载预训练模型
    base = build_transformer_model(
        config_path = config_path,
        checkpoint_path = checkpoint_path,
        model = MODEL_TYPE,
        return_keras_model = False
    )
    
    cls_features = keras.layers.Lambda(
        lambda x: x[:, 0],
        name = 'cls-token'
    )(base.model.output)  # shape=[batch_size,768]
    token_embedding = keras.layers.Lambda(
        lambda x: x[:, 1:-1],
        name = 'all-token'
    )(base.model.output)  #
    
    cnn1 = SetLearningRate(keras.layers.Conv1D(
        256,  # [[0.1,0.2],[0.3,0.1],[0.4,0.2]],[[0.12,0.32],[0.31,0.12],[0.24,0.12]]
        3,
        strides = 1,
        padding = 'same',  # 'valid'
        activation = 'relu',
        kernel_initializer = "he_normal"
    ), 20, True)(token_embedding)  # shape=[batch_size,maxlen-2,256]
    cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]
    
    cnn2 = SetLearningRate(keras.layers.Conv1D(
        256,
        4,
        strides = 1,
        padding = 'same',
        activation = 'relu',
        kernel_initializer = "he_normal"
    ), 20, True)(token_embedding)
    cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)
    
    cnn3 = SetLearningRate(keras.layers.Conv1D(
        256,
        5,
        strides = 1,
        padding = 'same',
        kernel_initializer = "he_normal"
    ), 20, True)(token_embedding)
    cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)
    
    cnn_features = keras.layers.concatenate(
        [cnn1, cnn2, cnn3],
        axis = -1)  # [batch_size,256*3]
    
    concat_features = keras.layers.concatenate(
        [cls_features, cnn_features],
        axis = -1)
    
    concat_features = keras.layers.Dropout(0.2)(concat_features)
    
    dense = SetLearningRate(keras.layers.Dense(
        units = 256,
        activation = 'relu',
        kernel_initializer = 'he_normal'
    ), 20, True)(concat_features)
    
    outputs = SetLearningRate(keras.layers.Dense(
        units = class_nums,
        activation = 'sigmoid',  # 多分类模型变多标签模型 softmax --> sigmoid
        kernel_initializer = 'he_normal'
    ), 20, True)(dense)
    
    model = keras.models.Model(base.model.input, outputs)
    # model.summary()
    return model


model = get_model()
plot_model(model, to_file = './model.jpg', show_shapes = True)
