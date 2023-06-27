import keras.preprocessing.image

from preprocessing import *
import tensorflow as tf

from draw_roc_curve import *

# model_Xception- last_conv_layer, Efficient = top_conv, MobileNet = conv_pw_13_relu

num_classes=1
learning_rate = 0.001
batch_size = 32 ##64
epochs = 100

X_train, y_train = train_set()
X_test, y_test = test_set()
X1_val, y1_val = valid_set()


# Define inputs

def Ensemble_model():

    ############### MobileNet ################

    inputs = tf.keras.layers.Input((IMG_WIDTH,IMG_HEIGHT, IMG_CHANNEL), name='input')

    def batch_and_ReLU(x):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    for i in range(5):
        x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(x)
        x = batch_and_ReLU(x)
        x = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same')(x)
        x = batch_and_ReLU(x)

    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(1024, (1, 1), strides=(1, 1), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.ZeroPadding2D()(x)

    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same')(x)
    x = batch_and_ReLU(x)
    x = tf.keras.layers.Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='conv_pw_13_relu')(x)  # 원래 1024
    x = batch_and_ReLU(x)
    x = tf.keras.layers.ZeroPadding2D()(x)  # zero padding 추가
    x = tf.keras.layers.BatchNormalization()(x)
    MobileNet_last_layer = tf.keras.layers.ReLU()(x)

    ################# Xception ####################

    ##### Entry flow #####
    e = tf.keras.layers.Conv2D(32, (3,3), strides=(2,2), padding='same')(inputs)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.ReLU()(e)
    e = tf.keras.layers.Conv2D(64,(3,3),strides=(1,1), padding='same')(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.ReLU()(e)

    ####conv1
    residual = tf.keras.layers.Conv2D(128, (1,1), strides=(2,2), padding='same')(e)

    e = tf.keras.layers.SeparableConv2D(128,(3,3), padding='same')(e)
    e = tf.keras.layers.ReLU()(e)
    e = tf.keras.layers.SeparableConv2D(128,(3,3), padding='same')(e)
    e = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(e)

    e = tf.keras.layers.Add()([residual, e])

    ####conv2
    residual = tf.keras.layers.Conv2D(256, (1,1), strides=(2,2), padding='same')(e)
    e = tf.keras.layers.ReLU()(e)
    e = tf.keras.layers.SeparableConv2D(256, (3,3), padding = 'same')(e)
    e = tf.keras.layers.ReLU()(e)
    e = tf.keras.layers.SeparableConv2D(256, (3,3), padding='same')(e)
    e = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(e)

    e = tf.keras.layers.Add()([residual, e])

    ####conv3
    residual = tf.keras.layers.Conv2D(728, (1,1), strides=(2,2), padding='same')(e)

    e = tf.keras.layers.ReLU()(e)
    e = tf.keras.layers.SeparableConv2D(728, (3,3), padding='same')(e)
    e = tf.keras.layers.ReLU()(e)
    e = tf.keras.layers.SeparableConv2D(728, (3,3), padding='same')(e)
    e = tf.keras.layers.MaxPooling2D((3,3), strides = (2,2), padding='same')(e)
    e = tf.keras.layers.Add()([residual, e])

    ##### Middle flow #####
    for i in range(8):

        residual = e

        m = tf.keras.layers.ReLU()(e)
        m = tf.keras.layers.SeparableConv2D(728, (3,3), padding='same')(m)
        m = tf.keras.layers.ReLU()(m)
        m = tf.keras.layers.SeparableConv2D(728, (3,3), padding='same')(m)
        m = tf.keras.layers.ReLU()(m)
        m = tf.keras.layers.SeparableConv2D(728, (3,3), padding='same')(m)

        m = tf.keras.layers.Add()([residual, m])

    ##### Exit flow #####

    residual = tf.keras.layers.Conv2D(1024,(1,1), strides=(2,2), padding='same')(m)

    ex = tf.keras.layers.ReLU()(m)
    ex = tf.keras.layers.SeparableConv2D(728, (3,3), padding='same')(ex)

    ex = tf.keras.layers.ReLU()(ex)
    ex = tf.keras.layers.SeparableConv2D(1024, (3,3), padding='same')(ex)
    ex = tf.keras.layers.MaxPooling2D((3,3), strides = (2,2), padding='same')(ex)

    ex = tf.keras.layers.Add()([residual, ex])
    ##
    ex = tf.keras.layers.SeparableConv2D(1536, (3,3), padding= 'same')(ex)
    ex = tf.keras.layers.ReLU()(ex)
    ex = tf.keras.layers.SeparableConv2D(2048, (3,3), padding='same', name = "last_conv_layer")(ex)     #원래 2048
    Xception_last_layer = tf.keras.layers.ReLU()(ex)

    # print(ex.shape, "Xception")        #(None, 7,7,2048)

    ############### EfficientNet ##################

    model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None,
                                                 input_tensor=(inputs))
    last_layer = model.get_layer('top_conv')
    EfficientNet_last_layer = last_layer.output  # (None,7,7,1280)


    ############### Concatenate layers ################
    x = tf.keras.layers.concatenate([Xception_last_layer, EfficientNet_last_layer, MobileNet_last_layer])        #(None, 3*channel)
    x = tf.keras.layers.Conv2D(2048, (1,1), padding='same', name='ensemble_model_last_layer')(x)
    x = tf.keras.layers.ReLU()(x)


    # New layer
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    # Build model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),  # accuracy
        keras.metrics.Precision(name="precision"),  # precision = specificity
        keras.metrics.Recall(name="recall"),  # recall = sensitivity
        keras.metrics.AUC(name='AUC')
    ])

    model.summary()
    return model

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


    _, accuracy, precision, recall, AUC = model.evaluate(X_test, y_test)
    pred_y = model.predict(X_test)
    return draw_roc_curve(y_test, pred_y)
#
# train_model(Ensemble_model())