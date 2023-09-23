import pydicom
import tensorflow as tf


def parse_dcm_image(img_path):
    """Parse a DICOM image and return it as a tensor."""
    img = pydicom.dcmread(img_path).pixel_array
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=-1)  # Add channel dimension
    return img
