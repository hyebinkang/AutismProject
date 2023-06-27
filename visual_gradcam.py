import keras.preprocessing.image

import model_Ensemble
import model_Mobilenet
import model_Xception
from model_Xception import Xception
from model_Mobilenet import MobileNet
from model_EfficientNet import EfficientNet
from model_Ensemble import Ensemble_model
from preprocessing import *
import tensorflow as tf
from IPython.display import Image, display
import matplotlib.cm as cm
from draw_roc_curve import *

#data 224*224
def get_img_array(img_path, size):

    img = keras.preprocessing.image.load_img(img_path, target_size=size)

    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 224, 224, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    superimposed_img.show(cam_path)

    # Display Grad CAM
    display(Image(cam_path))


#
last_conv_layer_name = 'last_conv_layer'                #model_Xception- last_conv_layer, Efficient = top_conv, MobileNet = conv_pw_13_relu, Ensemble =ensemble_model_last_layer


name_list = [145]                                                  #이미지 번호
for num in name_list:
    model = model_Xception.Xception()                                            #모델 가져오기
    img_path = autism_face_name[num]
    img_array = get_img_array(img_path, (IMG_WIDTH, IMG_HEIGHT))
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None)
    save_and_display_gradcam(img_path, heatmap, "./Cam_Xception/Xception_Autism_image{}".format(num), alpha=0.8)           #이미지 저장 주소 확인
