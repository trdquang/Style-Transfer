

"""# tf_hub"""

# Bước 1: Cài đặt các thư viện cần thiết
!pip install tensorflow tensorflow_hub

# Bước 2: Import các thư viện
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from io import BytesIO
from google.colab import files

# Hàm để tải và chuẩn bị ảnh
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Hàm để hiển thị ảnh
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

# Bước 3: Tải ảnh từ thiết bị lên
print("Upload content image")
content_image_path = list(files.upload().keys())[0]
print("Upload style image")
style_image_path = list(files.upload().keys())[0]

content_image = load_img(content_image_path)
style_image = load_img(style_image_path)

# Hiển thị ảnh đã tải
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# Bước 4: Tải mô hình style transfer từ TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Bước 5: Thực hiện style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Hiển thị kết quả
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 3, 2)
imshow(style_image, 'Style Image')

plt.subplot(1, 3, 3)
imshow(stylized_image, 'Stylized Image')
plt.show()

"""#end

"""

