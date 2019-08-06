# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import argparse
from flyai.dataset import Dataset
import keras
from keras.applications.resnet50 import ResNet50
from path import MODEL_PATH, DATA_PATH
from flyai.utils import remote_helper
import numpy as np
import random
import os
import cv2

# 判断路径是否存在
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
KERAS_MODEL_DIR = os.path.join(MODEL_PATH, "best.h5")

label_list = ['apron', 'bare-land', 'baseball-field', 'basketball-court', 'beach', 'bridge', 'cemetery', 'church', 'commercial-area', 'desert',
              'dry-field', 'forest', 'golf-course', 'greenhouse', 'helipad', 'ice-land', 'island', 'lake', 'meadow', 'mine',
              'mountain', 'oil-field', 'paddy-field', 'park', 'parking-lot', 'port', 'railway', 'residential-area', 'river', 'road',
              'roadside-parking-lot', 'rock-land', 'roundabout', 'runway', 'soccer-field',
              'solar-power-plant', 'sparse-shrub-land', 'storage-tank', 'swimming-pool', 'tennis-court',
              'terraced-field', 'train-station', 'viaduct', 'wind-turbine', 'works']

label_num = len(label_list)
img_size = [512, 512]

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
args = parser.parse_args()


print('epoch is %d, batch is %d'%(args.EPOCHS, args.BATCH))
# 加载全部数据,并且重新划分数据
dataset = Dataset()
x_train, y_train, x_val, y_val = dataset.get_all_data()
all_x = np.concatenate([x_train, x_val])
all_y = np.concatenate([y_train, y_val])
val_rate = 0.1
train_size = len(all_x)*(1-val_rate)
train_x = all_x[:train_size]
train_y = all_y[:train_size]
val_x = all_x[train_size:]
val_y = all_y[train_size:]
steps_per_epoch = int(train_x.shape[0] / args.BATCH)
print('train len is %d, val len is %d, all step is %d'%(train_x.shape[0], val_x.shape[0], steps_per_epoch))

# 逐批读取数据
def gen_batch_data(train_x, train_y, batch_size):
    num = train_x.shape[0]
    while True:
        random_index = [random.randint(0, num-1) for i in range(batch_size)]
        # 加载数据
        batch_x = []
        batch_y = []
        for i in random_index:
            img_path = os.path.join(DATA_PATH, train_x[i]['img_path'])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size[0], img_size[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            batch_x.append(img)

            label_onehot = np.zeros((label_num), dtype=int)
            index = label_list.index(train_y[i]['label'])
            label_onehot[index] = 1
            batch_y.append(label_onehot)
        yield np.array(batch_x), np.array(batch_y)

resnet = ResNet50(weights=None, include_top=False, pooling='avg')
path = remote_helper.get_remote_date("https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
resnet.load_weights(path)

x = resnet.output
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(45, activation='softmax')(x)
my_model = keras.models.Model(inputs=resnet.input, outputs=x)
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint(KERAS_MODEL_DIR, monitor='val_loss', verbose=1, mode='min')
earlystop = keras.callbacks.EarlyStopping(patience=5)
callbacks = [checkpoint, earlystop]
trian_gen = gen_batch_data(train_x, train_y, args.BATCH)
val_gen = gen_batch_data(val_x, val_y, args.BATCH)
my_model.fit_generator(generator=trian_gen, steps_per_epoch=steps_per_epoch, epochs=args.EPOCHS,
                       validation_data=val_gen, validation_steps=5, callbacks=callbacks)