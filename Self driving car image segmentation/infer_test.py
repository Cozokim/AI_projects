# Libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from seg_functions import infer

# Parameters
file_name = 'munster_001.png'
input_fname = 'static/images/'+file_name
target_fname = 'static/masks/'+file_name
img_height = 256
img_width = 512
n_classes = 8
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


# Load model
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

model = tf.keras.models.load_model('model/vgg_unet.h5', 
                                   custom_objects={'dice_coeff':dice_coeff,
                                                   'accuracy':'accuracy'})


image = img_to_array(load_img(input_fname,
                              target_size=(img_height, img_width)))/255
print(type(image), image.shape)

mask = img_to_array(load_img(target_fname,
                             target_size=(img_height, img_width),
                             color_mode = "grayscale"))
mask = np.squeeze(mask)

seg_img = infer(model=model, inp=input_fname, out_fname='./static/outputs/prediction.png', 
                  n_classes=n_classes, colors=class_colors,
                  prediction_width=512, prediction_height=256,
                  read_image_type=1)

print(type(seg_img), seg_img.shape)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1, 3, 1)
ax.set_title('Image')
ax.imshow(image)

ax1 = fig.add_subplot(1, 3, 2)
ax1.set_title('True mask')
ax1.imshow(mask, cmap='nipy_spectral_r')

ax2 = fig.add_subplot(1, 3, 3)
ax2.set_title('predicted_Mask')
ax2.imshow(seg_img, cmap='nipy_spectral_r')
plt.show()