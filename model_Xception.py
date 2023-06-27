import keras.preprocessing.image
from preprocessing import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from draw_roc_curve import *
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


num_classes = 1
learning_rate = 0.001
batch_size = 32
epochs = 100

X_train, y_train = train_set()
X_test, y_test = test_set()
X1_val, y1_val = valid_set()

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))    # 299*299*3

def Xception():
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
    ex = tf.keras.layers.ReLU()(ex)
    ex = tf.keras.layers.GlobalAvgPool2D()(ex)

    print(ex.shape)        #(None, 7,7,2048)


    outputs = tf.keras.layers.Dropout(0.5)(ex)
    outputs = tf.keras.layers.Dense(2048)(outputs)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(outputs)


    model = tf.keras.models.Model(inputs = inputs, outputs = outputs, name='Xception')


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[
                    keras.metrics.BinaryAccuracy(name="accuracy"),                                  #accuracy
                    keras.metrics.Precision(name="precision"),                                      #precision = specificity
                    keras.metrics.Recall(name="recall"),                                            #recall = sensitivity
                    keras.metrics.AUC(name='AUC')
                      ]
                  )
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

    # history = model.fit(X_train, y_train, validation_split = 0.1, batch_size=batch_size, epochs=epochs, callbacks = callbacks)
    _, accuracy, precision, recall, AUC = model.evaluate(X_test, y_test)
    pred_y = model.predict(X_test)

    return draw_roc_curve(y_test, pred_y)

# def get_img_array(img_path, size):
#
#     img = keras.preprocessing.image.load_img(img_path, target_size=size)
#
#     array = keras.preprocessing.image.img_to_array(img)
#     # We add a dimension to transform our array into a "batch"
#     # of size (1, 224, 224, 3)
#     array = np.expand_dims(array, axis=0)
#     return array
#
#
# def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
#     # First, we create a model that maps the input image to the activations
#     # of the last conv layer as well as the output predictions
#     grad_model = tf.keras.models.Model(
#         [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#     )
#
#     # Then, we compute the gradient of the top predicted class for our input image
#     # with respect to the activations of the last conv layer
#     with tf.GradientTape() as tape:
#         last_conv_layer_output, preds = grad_model(img_array)
#         if pred_index is None:
#             pred_index = tf.argmax(preds[0])
#         class_channel = preds[:, pred_index]
#
#     # This is the gradient of the output neuron (top predicted or chosen)
#     # with regard to the output feature map of the last conv layer
#     grads = tape.gradient(class_channel, last_conv_layer_output)
#
#     # This is a vector where each entry is the mean intensity of the gradient
#     # over a specific feature map channel
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#
#     # We multiply each channel in the feature map array
#     # by "how important this channel is" with regard to the top predicted class
#     # then sum all the channels to obtain the heatmap class activation
#     last_conv_layer_output = last_conv_layer_output[0]
#     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#
#     # For visualization purpose, we will also normalize the heatmap between 0 & 1
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#
#     return heatmap.numpy()
#
#
# def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
#     # Load the original image
#     img = keras.preprocessing.image.load_img(img_path)
#     img = keras.preprocessing.image.img_to_array(img)
#
#     # Rescale heatmap to a range 0-255
#     heatmap = np.uint8(255 * heatmap)
#
#     # Use jet colormap to colorize heatmap
#     jet = cm.get_cmap("jet")
#
#     # Use RGB values of the colormap
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap]
#
#     # Create an image with RGB colorized heatmap
#     jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
#     jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
#     jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
#
#     # Superimpose the heatmap on original image
#     superimposed_img = jet_heatmap * alpha + img
#     superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
#
#     # Save the superimposed image
#     superimposed_img.save(cam_path)
#     superimposed_img.show(cam_path)
#
#     # Display Grad CAM
#     display(Image(cam_path))
#
#
#
# model_builder = keras.applications.xception.Xception
# img_size = (224,224)
# preprocess_input = keras.applications.xception.preprocess_input
# decode_predictions = keras.applications.xception.decode_predictions
#
# last_conv_layer_name = 'last_conv_layer'
#
# img_path = "./AutismDataset/test/Autistic.28.jpg"
#
#
# model = Xception()
#
# # Remove last layer's softmax
# model.layers[-1].activation = None
# img_array = preprocess_input(get_img_array(img_path, size=img_size))
# # Print what the top predicted class is
# preds = model.predict(img_array)
# # print("Predicted:", decode_predictions(preds, top=1)[0])
#
# # Generate class activation heatmap
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
#
# # Display heatmap
# plt.matshow(heatmap)
# plt.show()
# save_and_display_gradcam(img_path, heatmap, "./Autism_image{}.jpg", alpha=0.8)
