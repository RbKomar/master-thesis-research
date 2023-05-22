import tensorflow as tf

def resize_image(image, size):
    """Resize an image to the given size."""
    resized_image = tf.image.resize(image, size)
    return resized_image

def random_crop(image, size):
    """Randomly crop an image to the given size."""
    cropped_image = tf.image.random_crop(image, size)
    return cropped_image

def random_flip_left_right(image):
    """Randomly flip an image horizontally (left to right)."""
    flipped_image = tf.image.random_flip_left_right(image)
    return flipped_image

def random_brightness(image, max_delta):
    """Randomly adjust the brightness of an image."""
    brightness_adjusted_image = tf.image.random_brightness(image, max_delta=max_delta)
    return brightness_adjusted_image

@tf.function
def preprocess_image(image):
    """Preprocess an image for use in a deep learning model."""
    # Resize the image to a fixed input size
    image = resize_image(image, [224, 224])

    # Apply random data augmentation
    image = random_crop(image, [224, 224, 3])
    image = random_flip_left_right(image)
    image = random_brightness(image, max_delta=0.1)

    # Convert the image to float32 and scale the pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    return image
