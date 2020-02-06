from keras.models import load_model
import cv2 as cv
import os
import numpy as np

# 预测
model_path = r"E:\Practice_project\dogvscat_henry\weights\vgg_model.h5"
image_path = r"test_img"

img_name_list = os.listdir(image_path)
# 加载要预测的图像
img_list = np.array([])

for n in img_name_list:
    img = cv.imread(os.path.join(image_path, n))
    img = cv.resize(img, (100, 100))/255.0
    img = img[np.newaxis, :, :, :]
    if len(img_list.shape) > 3:

        img_list = np.concatenate((img_list, img), axis=0)

    else:
        img_list = img
# 加载模型
model = load_model(model_path)
pred = model.predict(img_list, batch_size=4)
# print(pred)
for index, i in enumerate(pred):
    maxer = np.argmax(i)

    # 猫0狗1
    if maxer == 0:
        print(img_name_list[index], "is a cat!")
        img = cv.imread(os.path.join(image_path, img_name_list[index]))
        cv.imshow("is a cat", img)
        cv.waitKey(0)

    else:
        print(img_name_list[index], "is a dog!")
        img = cv.imread(os.path.join(image_path, img_name_list[index]))
        cv.imshow("is a dog", img)
        cv.waitKey(0)

print("over!")