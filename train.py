from model import vgg_model
from processing_data import get_img_name_list, data_generator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# 训练
seed = 17   # 随机种子
h, w = 100, 100  # 图像的长宽
epoch = 4
batch_size = 64
learning_rate = 0.0001

# vgg16的权重地址
vgg_weights_path = r"E:\Practice_project\dogvscat_henry\weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
model_save_path = r"weights/vgg_model.h5"   # 模型的保存地址
data_path = r"E:\Dataset\dogvscat\train"    # 数据集的train文件夹

# 加载数据

x_name, y = get_img_name_list(data_path)
train_X, test_X, train_Y, test_Y = train_test_split(x_name, y, test_size=0.2, random_state=0)

np.random.seed(seed)
model = vgg_model(vgg_weights_path)
# adam = Adam(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=Adam(lr=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit_generator(generator=data_generator(train_X, train_Y, batch_size, h, w),
                    steps_per_epoch=len(train_X) // batch_size,
                    epochs=epoch, verbose=1, validation_data=data_generator(test_X, test_Y, h, w),
                    validation_steps=len(test_X) // batch_size)
# 保存权重
# model.save(model_save_path)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()