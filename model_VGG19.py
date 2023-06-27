import keras.preprocessing.image

from preprocessing import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from draw_roc_curve import *
from visual_gradcam import *

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))
print(inputs.shape)

batch_size = 32
epochs = 40

X_train, y_train = train_set()
X_test, y_test = test_set()
X1_val, y1_val = valid_set()
# X_train, X_test, y_train, y_test = train_test_split(X1_train, y1_train, test_size=0.2)

x1 = tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu')(inputs)                #224*224*3
x1 = tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu')(x1)
m1 = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(x1)

x2 = tf.keras.layers.Conv2D(128,(3,3), padding='same', activation='relu')(m1)
x2 = tf.keras.layers.Conv2D(128,(3,3), padding='same', activation='relu')(x2)
m2 = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(x2)

x3 = tf.keras.layers.Conv2D(256,(3,3), padding='same', activation='relu')(m2)
x3 = tf.keras.layers.Conv2D(256,(3,3), padding='same', activation='relu')(x3)
x3 = tf.keras.layers.Conv2D(256,(3,3), padding='same', activation='relu')(x3)
x3 = tf.keras.layers.Conv2D(256,(3,3), padding='same', activation='relu')(x3)
m3 = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(x3)

x4 = tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu')(m3)
x4 = tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu')(x4)
x4 = tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu')(x4)
x4 = tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu')(x4)
m4 = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(x4)

x5 = tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu')(m4)
x5 = tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu')(x5)
x5 = tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu')(x5)
x5 = tf.keras.layers.Conv2D(512,(3,3), padding='same', activation='relu', name='block5_conv4')(x5)
m5 = tf.keras.layers.MaxPooling2D((2,2), strides=(2,2))(x5)

x = tf.keras.layers.Flatten()(m5)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)


model = tf.keras.models.Model(inputs = inputs, outputs=outputs)
# model = tf.keras.applications.VGG19(include_top=False, weights=None, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL), classes=1, classifier_activation='sigmoid')


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


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),                                  #accuracy
                keras.metrics.Precision(name="precision"),                                      #precision = specificity
                keras.metrics.Recall(name="recall"),                                            #recall = sensitivity
                keras.metrics.AUC(name='AUC')
                ]
              )

model.summary()

checkpoint_filepath = 'checkpoint.h5'


callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, verbose=1, save_best_only=True)
]

history = model.fit(train_generator, steps_per_epoch=len(X_train)//batch_size, validation_data = val_generator, validation_steps=len(X1_val)//batch_size,
                    batch_size=batch_size, epochs=epochs, callbacks = callbacks)

_, accuracy, precision, recall, AUC = model.evaluate(X_test, y_test)
pred_y = model.predict(X_test)
draw_roc_curve(y_test, pred_y)

img_path = autism_face_name[0]

preprocess_input = keras.applications.vgg19.preprocess_input

# 모델의 출력 레이어 가져오기
last_conv_layer_name = 'block5_conv4'
classifier_layer = model.layers[-1]

img_array = preprocess_input(get_img_array(img_path, (IMG_WIDTH, IMG_HEIGHT)))

# 마지막 conv 레이어 가져오기
last_conv_layer = model.get_layer(last_conv_layer_name)

preds = model.predict(img_array)

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

plt.matshow(heatmap)
plt.show()

save_and_display_gradcam(img_path, heatmap, "Cam_VGG.jpg", alpha=0.4)