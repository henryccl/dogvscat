import os
import numpy as np
import cv2 as cv
from keras.utils import to_categorical


def get_img_name_list(data_path):
    img_name_list = os.listdir(data_path)
    y = []
    x_name = []
    for f in img_name_list:
        x_name.append(os.path.join(data_path, f))
        # 猫0狗1
        if "cat" in f:
            y.append(0)
        else:
            y.append(1)
    y = np.array(y)
    x_name = np.array(x_name)
    return x_name, y


def data_generator(all_img_name, all_label, batch_size, h=100, w=100):
    """
    该函数用于生成批量的数据，用于fit_generator批量训练
    :param all_train_name:
    :param batch_size:
    :return:
    """

    batches = len(all_img_name) // batch_size

    while True:
        for i in range(batches):
            name_batch = all_img_name[i * batch_size: (i + 1) * batch_size]
            label_batch = all_label[i * batch_size: (i + 1) * batch_size]
            # label 转化为one-hot编码
            Y = to_categorical(label_batch, num_classes=2)

            X = np.array([])
            for j in range(batch_size):
                img_path = name_batch[j]
                labels = label_batch

                # 读取img
                img = cv.imread(img_path)
                # resize
                img = cv.resize(img, (h, w))/255.0

                if len(X.shape) < 3:
                    X = img[np.newaxis, :, :]
                else:
                    X = np.concatenate((X, img[np.newaxis, :, :]), axis=0)

            yield (X, Y)


if __name__ == "__main__":
    data_path = r"E:\Dataset\dogvscat\train"
    x_name, y = get_img_name_list(data_path)
    index = np.arange(0, len(x_name))
    np.random.shuffle(index)
    x_name = x_name[index]
    y = y[index]
    print(x_name, y)
    gen = data_generator(x_name, y, batch_size=6, h=100, w=100)

    X, Y = next(gen)

    pass