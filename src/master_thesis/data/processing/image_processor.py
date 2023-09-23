import numpy as np
import tensorflow as tf


class ImageProcessor:
    def parse_image(self, img_path, obscure_images_percent):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)  # assuming images are in JPEG format
        image = tf.image.resize(image, (224, 224))  # assuming resizing to 224x224 for model input
        if obscure_images_percent > 0:
            image = self._obscure_image(image, obscure_images_percent)
        image = tf.cast(image, tf.float32) / 255.0  # Normalization to [0,1]
        return image

    @staticmethod
    def _obscure_image(image, obscure_images_percent):
        img = image.numpy()
        total_area = img.shape[0] * img.shape[1]
        cover_area = total_area * obscure_images_percent
        cover_rows = int(np.sqrt(cover_area))
        img[:cover_rows, :cover_rows] = 0
        return tf.convert_to_tensor(img)

    @staticmethod
    def occlude_image(image, occlude_percent=0.1):
        img = image.numpy()
        total_area = img.shape[0] * img.shape[1]
        occlude_area = int(total_area * occlude_percent)

        for _ in range(occlude_area):
            row = np.random.randint(0, img.shape[0])
            col = np.random.randint(0, img.shape[1])
            img[row, col] = 0  # occlude the pixel

        return tf.convert_to_tensor(img)

    @staticmethod
    def add_noise(image, noise_level=0.01):
        img = image.numpy()
        noise = np.random.normal(0, noise_level, img.shape)
        img = np.clip(img + noise, 0, 1)  # add noise and clip to valid range
        return tf.convert_to_tensor(img)

    @staticmethod
    def remove_section(image, section=(slice(50, 100), slice(50, 100))):
        img = image.numpy()
        img[section] = 0  # remove the section
        return tf.convert_to_tensor(img)
