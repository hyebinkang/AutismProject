# import tensorflow as tf
# from preprocessing import *
#
# class MBConv(tf.keras.layers.Layer):
#     def __init__(self, input_channels, output_channels, kernel_size, strides, expand_ratio, se_ratio, dropout_rate):
#         super(MBConv, self).__init__()
#         self.expand_ratio = expand_ratio
#         self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
#         self.dropout_rate = dropout_rate
#
#         self.conv1 = tf.keras.layers.Conv2D(
#             filters=input_channels * expand_ratio,
#             kernel_size=1,
#             padding='same',
#             use_bias=False,
#             activation=None
#         )
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.dwconv = tf.keras.layers.DepthwiseConv2D(
#             kernel_size=kernel_size,
#             strides=strides,
#             padding='same',
#             use_bias=False,
#             activation=None
#         )
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         self.se = self._build_se_layer(filters=input_channels * expand_ratio, se_ratio=se_ratio)
#         self.conv2 = tf.keras.layers.Conv2D(
#             filters=output_channels,
#             kernel_size=1,
#             padding='same',
#             use_bias=False,
#             activation=None
#         )
#         self.bn3 = tf.keras.layers.BatchNormalization()
#
#         if strides == 1 and input_channels == output_channels:
#             self.skip = tf.keras.layers.Activation('linear')
#         else:
#             self.skip = tf.keras.layers.Conv2D(
#                 filters=output_channels,
#                 kernel_size=1,
#                 padding='same',
#                 use_bias=False,
#                 activation=None
#             )
#         self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
#
#     def call(self, inputs, training=False):
#         x = inputs
#         if self.expand_ratio != 1:
#             x = self.conv1(x)
#             x = self.bn1(x, training=training)
#             x = tf.nn.swish(x)
#
#         x = self.dwconv(x)
#         x = self.bn2(x, training=training)
#         x = tf.nn.swish(x)
#
#         if self.has_se:
#             x = self.se(x)
#
#         x = self.conv2(x)
#         x = self.bn3(x, training=training)
#
#         if self.skip is not None:
#             skip = self.skip(inputs)
#         else:
#             skip = inputs
#
#         x = x + skip
#         x = tf.nn.swish(x)
#         x = self.dropout(x, training=training)
#
#         return x
#
#     def _build_se_layer(self, filters, se_ratio):
#         filters_se = max(1, int(filters * se_ratio))
#         se = tf.keras.Sequential([
#             tf.keras.layers.GlobalAveragePooling2D(),
#             tf.keras.layers.Dense(filters_se, activation='swish', use_bias=True),
#             tf.keras.layers.Dense(filters, activation='sigmoid', use_bias=True),
#         ], name='se_block')
#         return se
#
#
# model = tf.keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=None,
#                      input_shape=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNEL), pooling=None, classes=1, classifier_activation='sigmoid')
#
# model = tf.keras.models.Sequential()
# model.add(model)
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation = 'relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='Adam',                                 #adam ?
#               metrics=['accuracy'])
#
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
#     tf.keras.callbacks.TensorBoard(log_dir='logs')
# ]
#
# history = model.fit(X_train, y_train, validation_data=(X1_val, y1_val), batch_size=32, epochs=100, callbacks = callbacks)

from model_EfficientNet import *
from model_Xception import *
from model_Mobilenet import *
