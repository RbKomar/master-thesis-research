import tensorflow as tf
import numpy as np
import unittest

from _old.preprocessing import preprocess_image


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_image(self):
        # Load a test image
        test_image = tf.constant(np.random.randint(0, 256, size=(480, 640, 3)).astype(np.uint8))

        # Preprocess the test image using the preprocessing function
        preprocessed_image = preprocess_image(test_image)

        # Check that the preprocessed image has the correct shape
        self.assertEqual(preprocessed_image.shape, (224, 224, 3))

        # Check that the preprocessed image has the correct data type
        self.assertEqual(preprocessed_image.dtype, tf.float32)

        # Check that the preprocessed image has been normalized correctly
        self.assertTrue(tf.reduce_all(preprocessed_image >= 0.0))
        self.assertTrue(tf.reduce_all(preprocessed_image <= 1.0))


if __name__ == '__main__':
    unittest.main()
