import keras.preprocessing.image

from preprocessing import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from draw_roc_curve import *
from multiprocessing import Pool
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# print(inputs.shape, "input")
#
num_classes = 1

learning_rate = 0.001
batch_size = 32
epochs = 100

X_train, y_train = train_set()
X_test, y_test = test_set()
X1_val, y1_val = valid_set()

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))

def MobileNet():
    def batch_and_ReLU(x):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    x = tf.keras.layers.Conv2D(32,(3,3), strides=(2,2), padding='same')(inputs)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1,1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(64, (1,1), strides=(1,1), padding='same')(x)
    x = batch_and_ReLU(x)
    # x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3,3), strides=(2,2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(128, (1,1), strides=(1,1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.DepthwiseConv2D((3,3),strides=(2,2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(128, (1,1), strides = (1,1), padding = 'same')(x)
    x = batch_and_ReLU(x)
    # x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3,3), strides=(2,2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(256, (1,1), strides = (1,1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.DepthwiseConv2D((3,3), strides = (1,1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), padding='same')(x)
    x = batch_and_ReLU(x)
    # x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3,3), strides=(2,2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(512, (1,1), strides = (1,1), padding='same')(x)
    x = batch_and_ReLU(x)
    # x = tf.keras.layers.ZeroPadding2D()(x)
    for i in range(5):
        x = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1,1), padding='same')(x)
        x = batch_and_ReLU(x)
        x = tf.keras.layers.Conv2D(512, (1,1), strides=(1,1), padding='same')(x)
        x = batch_and_ReLU(x)

    # x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3,3), strides = (2,2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(1024, (1,1), strides=(1,1), padding='same')(x)
    x = batch_and_ReLU(x)
    # x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3,3), strides=(2,2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(1024, (1,1), strides=(1,1), padding='same', name='conv_pw_13_relu')(x)       #원래 1024
    x = batch_and_ReLU(x)

    outputs = tf.keras.layers.AveragePooling2D(pool_size=(7,7), padding='same')(x)
    outputs = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Flatten()(outputs)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(outputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),  # accuracy
        keras.metrics.Precision(name="precision"),  # precision = specificity
        keras.metrics.Recall(name="recall"),  # recall = sensitivity
        keras.metrics.AUC(name='AUC')
    ]
                  )

    model.summary()

    return model
    ## application방법
    # model = tf.keras.applications.MobileNet(include_top=True, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), classes=1, classifier_activation='sigmoid')


def train_model(model):
    train_datagen = ImageDataGenerator(
        rotation_range=30,  # 이미지 회전 각도 범위
        width_shift_range=0.2,  # 이미지 가로 이동 범위 (비율)
        height_shift_range=0.2,  # 이미지 세로 이동 범위 (비율)
        shear_range=0.2,  # 이미지 전단 강도 범위
        zoom_range=0.2,  # 이미지 확대/축소 범위
        horizontal_flip=True,  # 좌우 반전
        vertical_flip=False,  # 상하 반전
        fill_mode='nearest'  # 이미지를 이동/회전 시 생기는 공간을 어떻게 채울 것인지
    )

    train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)

    val_datagen = ImageDataGenerator(rescale = 1./255)
    val_generator = val_datagen.flow(X1_val, y1_val, batch_size = batch_size)

    checkpoint_filepath = 'checkpoint.h5'


    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True)
    ]

    history = model.fit(train_generator, steps_per_epoch=len(X_train)//batch_size, validation_data = val_generator, validation_steps=len(X1_val)//batch_size,
                        batch_size=batch_size, epochs=epochs, callbacks = callbacks)


    # history = model.fit(X_train, y_train, validation_split = 0.1, batch_size=batch_size, epochs=epochs, callbacks = callbacks)
    _, accuracy, precision, recall, AUC = model.evaluate(X_test, y_test)
    pred_y = model.predict(X_test)
    return draw_roc_curve(y_test, pred_y)

# train_model(MobileNet())

