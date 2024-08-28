import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
cv2.ocl.setUseOpenCL(False)
import warnings
warnings.filterwarnings('ignore')

# Make sure that the train image is the image that will be transformed
train_photo = cv2.imread('imagetwo.jpg')

# OpenCV defines the color channel in the order BGR
# Hence converting to RGB for Matplotlib
train_photo = cv2.cvtColor(train_photo,cv2.COLOR_BGR2RGB)

# converting to grayscale
train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)

# Do the same for the query image
query_photo = cv2.imread('imageone.jpg')
query_photo = cv2.cvtColor(query_photo,cv2.COLOR_BGR2RGB)
query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

# Now view/plot the images
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
ax1.imshow(query_photo, cmap="gray")
ax1.set_xlabel("Image 1", fontsize=14)

ax2.imshow(train_photo, cmap="gray")
ax2.set_xlabel("Image 2", fontsize=14)

#plt.savefig('.jpeg', bbox_inches='tight', dpi=300, format='jpeg')

plt.show()
import os

import tensorflow as tf
from tensorflow import keras
new_model = tf.keras.models.load_model('my_denoise_model.keras')

# Show the model architecture
new_model.summary()

def plot_results(noise_image, reconstructed_image, image):
    w = 15
    h = len(noise_image)*5
    fig = plt.figure(figsize=(w, h))
    columns = 3
    rows = len(noise_image)
    for i in range(1, rows*columns, columns):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.title('Image with noise')
        plt.imshow(images[int((i-1)/columns)])

        fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        plt.title('Reconstructed Image')
        plt.imshow(reconstructed[int((i-1)/columns)])

        fig.add_subplot(rows, columns, i+2)
        plt.axis('off')
        plt.title('Original Image')
        plt.imshow(images[int((i-1)/columns)])

    plt.show()



main_dir = 'predict_data/'
all_image_paths = ['imageone.jpg', 'imagetwo.jpg','ball.jpeg']

print(all_image_paths)
print('Total number of images:', len(all_image_paths))

from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt

def open_images(paths, size=2):
    '''
    Given an array of paths to images, this function opens those images,
    and returns them as an array of shape (None, Height, Width, Channels)
    '''
    images = []
    for path in paths:
        image = load_img(path, target_size=(size, size, 3))
        image = np.array(image)/255.0 # Normalize image pixel values to be between 0 and 1
        images.append(image)
    return np.array(images)

import random
paths = random.sample(all_image_paths, 3)
images = open_images(paths, size=224)
# Amount of noise = random value between 0.1 and 0.15
amount = random.uniform(0.1,0.15)
# noise_images = add_noise(images, amount=amount)
reconstructed = new_model.predict(images)
plot_results(images, reconstructed, images)