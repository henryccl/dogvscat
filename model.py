from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Flatten
from keras.models import Model
import numpy as np


def vgg_model(vgg_weights_path="imagenet"):
    # 定义模型
    base_model = VGG16(weights=vgg_weights_path,
                       include_top=False, input_shape=(100, 100, 3))

    x = Flatten()(base_model.output)
    x = Dense(1024, activation="relu")(x)
    x = Dense(200, activation="relu")(x)
    y_pred = Dense(2, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=y_pred)
    model.summary()

    return model

