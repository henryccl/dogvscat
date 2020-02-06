# dogvscat
  using keras implement vgg and then fine-tining
  https://blog.csdn.net/qq_41997888/article/details/104162436
# How to run?

### 1.change the path in train.py to your path for model training
  vgg_weights_path = r"E:\Practice_project\dogvscat_henry\weights\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
  model_save_path = r"weights/vgg_model.h5"  
  data_path = r"E:\Dataset\dogvscat\train"   

### 2.change the path in predict.py to your path for model prediction
  model_path = r"E:\Practice_project\dogvscat_henry\weights\vgg_model.h5"
  image_path = r"test_img"

### 3.python train.py
