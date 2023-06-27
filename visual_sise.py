import tensorflow as tf
from preprocessing import *
import numpy as np
import model_Xception
from model_Xception import *
import matplotlib.pyplot as plt

#모델의 layer 개수에 따른 multilayer output가져오기, len(model.get_layer)
#model = 모델 가져오기, 레이어 개수 가져오기

model = model_Xception.Xception()
layers = []
attribution_masks = []
attribution_masks_np = []
num_rows = 5
num_cols = 5
img = tf.keras.preprocessing.image.load_img(autism_face_name[0], target_size=(IMG_WIDTH, IMG_HEIGHT))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.keras.backend.constant(img_array)  # Tensor로 변환
blocks = tf.zeros_like(img_array)  # 초기값을 0으로 설정
num_layers= len(model.layers)
print(num_layers)
pred_index = None       #z클래스 인덱스 초기값

for i in range(num_layers):
    layer = model.get_layer(index=i)                                    # 모델의 레이어의 feature map을 가져오기
    layers.append(layer)

    with tf.GradientTape() as tape:                                     # 미분, 역전파
        conv_layer_output = layer(img_array)
        if pred_index is None:
            pred_index = tf.argmax(conv_layer_output[0]).numpy()
        class_channel = conv_layer_output[:, pred_index]                # 역전파

    backward = tape.gradient(class_channel, conv_layer_output)

    for i in range(len(layers)):
        gradients = backward[i]
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))            # 차원 감소하며 평균
        conv_output = conv_layer_output[i]
        attribution_mask = conv_output @ pooled_gradients[..., tf.newaxis]
        attribution_mask = tf.squeeze(attribution_mask)
        attribution_mask = tf.maximum(attribution_mask, 0) / tf.math.reduce_max(attribution_mask)
        attribution_masks.append(attribution_mask)

    for i in range(len(attribution_masks)):
        gradients = backward[i]
        attribution_mask = attribution_masks[i]
        weighted_attribution_mask = tf.reduce_mean(tf.multiply(gradients, attribution_mask), axis=(0,1,2))

block_size = 5

for row in range(0, IMG_HEIGHT-block_size+1, block_size):
    for col in range(0, IMG_WIDTH-block_size+1, block_size):
        block = blocks[row:row+block_size, col:col+block_size, :]
        block = tf.expand_dims(block, axis=0)
        block_output = model(block)
        block_features = block_output[0]
        block_features = tf.squeeze(block_features)
        attribution_masks[row//block_size, col//block_size, :] = block_features


fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10,10))

for row in range(num_rows):
    for col in range(num_cols):
        mask = attribution_masks[row, col]
        axs[row, col].imshow(mask)
        axs[row, col].axis('off')
