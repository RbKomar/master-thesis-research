import pydicom
import tensorflow as tf
from pydicom.errors import InvalidDicomError


def parse_dcm_image(img_path: str) -> tf.Tensor:
    """
    Parse a DICOM image and return it as a tensor with an added channel dimension.

    :param img_path: str, Path to the DICOM image file.
    :return: tf.Tensor, A TensorFlow tensor representing the DICOM image.
    """
    try:
        # Read DICOM image
        img = pydicom.dcmread(img_path).pixel_array
    except FileNotFoundError:
        raise FileNotFoundError(f"The DICOM file at {img_path} was not found.")
    except InvalidDicomError:
        raise ValueError(f"The file at {img_path} is not a valid DICOM file.")

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # Add a channel dimension to the tensor for compatibility with model input
    img = tf.expand_dims(img, axis=-1)

    return img
