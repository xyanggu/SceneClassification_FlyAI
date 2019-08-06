# -*- coding: utf-8 -*
import os
from flyai.model.base import Base
import keras
from path import MODEL_PATH
KERAS_MODEL_DIR = os.path.join(MODEL_PATH, "best.h5")


class Model(Base):
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = keras.models.load_model(KERAS_MODEL_DIR)

    '''
    评估一条数据
    '''

    def predict(self, **data):

        x_data = self.dataset.predict_data(**data)
        pred = self.model.predict(x_data)
        # 将预测数据转换成对应标签  to_categorys 会去调用 processor.py 中的 output_y 方法
        data = self.dataset.to_categorys(pred)
        return data

    '''
    评估的时候会调用该方法实现评估得分
    '''

    def predict_all(self, datas):

        labels = []
        for data in datas:
            # 获取需要预测的图像数据， predict_data 方法默认会去调用 processor.py 中的 input_x 方法
            x_data = self.dataset.predict_data(**data)
            pred = self.model.predict(x_data)
            # 将预测数据转换成对应标签  to_categorys 会去调用 processor.py 中的 output_y 方法
            data = self.dataset.to_categorys(pred)
            labels.append(data)
        return labels

