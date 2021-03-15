# example of horizontal shift image augmentation
from numpy import expand_dims
from matplotlib import pyplot
import tensorflow as tf


# load the image
img = tf.keras.preprocessing.image.load_img('bird.jpg')
# convert to numpy array
data = tf.keras.preprocessing.image.img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=[-0.1,-0.1],
    height_shift_range=[-0.1,-0.1],
    rotation_range=25,
    zoom_range=0.2
)
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
    # define subplot
    pyplot.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    pyplot.imshow(image)
# show the figure
pyplot.show()