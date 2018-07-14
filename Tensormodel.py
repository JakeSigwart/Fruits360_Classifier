import tensorflow as tf

#Returns the numeric predictions or the numeric classes
def Reduce_one_hot(y_norm):
    return tf.argmax(y_norm, 1)

#IMAGE FUNCTIONS

#Input: an image, the size of the crop square and number of channels
#Output: Cropped image
def crop_image(image, img_size_cropped, num_channels):
    image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])
    return image

#Flips an image about the diagonal (top-left to bottom-right) axis
def Image_transpose(image):
    image = tf.image.transpose_image(image)
    return image

#Input: A grayscale image w/ dimensions: [height, width, 1]
#Output: An RGB image w/ dimensions: [height, width, 3]
def Image_grayscale_to_rgb(image):
    image = tf.image.grayscale_to_rgb(image, name=None)
    return image

#Input: An RGB image w/ dimensions: [height, width, 3]
#Output: A grayscale image w/ dimensions: [height, width, 1]
def Image_rgb_to_grayscale(image):
    image = tf.image.rgb_to_grayscale(image)
    return image

#Image Pre-processing
#Input: an image, a scalar int representing the number of rotations CCW by 90 degrees
#Output: The image rotated
def Image_rotate(image, num_rot):
    image = tf.image.rot90(image, k=1, name=None)
    return image

#Input: A single RGB image: [height, width, 3]
#Output: The RGB image with randomly adjusted hue, contrast, brightness, saturation
def Image_random_rgb_distort(image):
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_brightness(image, max_delta=.2)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    #Make sure color levels remain in the range [0, 1]
    image = tf.minimum(image, 1.0)
    image = tf.maximum(image, 0.0)
    return image

#Input: Any image: [height, width, channels]
#Output: The same image with a 50% chance of being mirrored vertically and 50% horizontally
def Image_random_flip_distort(image):
    image = tf.image.random_flip_left_right(image)
    return image

#Randomly flip and adjust each of the input 
def pre_process_images(images, process_images):
    # Use TensorFlow to loop over all the input images
    #tf.map_fn: Unpack all images along dimension: 0
    #An optional 3rd parameter holds the max number of iterations allowed to run in parallel (default=10)
    def f1(): return images
    def f2(): return tf.map_fn(lambda image: Image_random_rgb_distort(image), images)
    def f3(): return tf.map_fn(lambda image: Image_random_flip_distort(image), images)
    images = tf.cond(process_images, f2, f1)
    images = tf.cond(process_images, f3, f1)
    return images

#Input: 4-D Tensor of shape [batch, height, width, channels] or 3-D tensor of shape [height, width, channels],  A 1-D int32 Tensor of 2 elements: new_height, new_width
#Output: Each image is resized to: [new_height, new_width, channels] 
def Resize_images_to(images, new_shape):
    images = tf.image.resize_images(images, new_shape)
    return images

#Input: 4-D Tensor of shape [batch, height, width, channels] or 3-D tensor of shape [height, width, channels],  [target_height,target_width]
#Output: Image resized by cropping or padding
def Resize_images_crop_pad(images, new_shape):
    images = tf.image.resize_images(images, new_shape)
    return images

#Get files
#Input: 0-D string. The encoded image bytes, target ouput color channels
#Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate operation to convert the input bytes string into a Tensor of type uint8.
#Output: gifs return a tensor of dim: [num_frames, height, width, 3]; PNG, JPEG dim: [height, width, num_channels]
def Decode_image_from_bytes(contents, output_color_channels=None):
    image = tf.image.decode_image(contents, channels=output_color_channels)
    return image

def Encode_image_as_jpeg(image):
    return tf.image.encode_jpeg(image)
        
def Encode_image_as_png(image):
    return tf.image.encode_png(image)