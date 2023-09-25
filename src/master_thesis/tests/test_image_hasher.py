from PIL import Image
from src.master_thesis.data.handler.hashing import ImageHasher
import unittest


class ImageHasherTest(unittest.TestCase):

    @staticmethod
    def _create_sample_images(duplicate=False) -> tuple[Image.Image, Image.Image]:
        if duplicate:
            img1 = Image.new('RGB', (60, 30), color='red')
            img2 = Image.new('RGB', (60, 30), color='red')
        else:
            img1 = Image.new('RGB', (60, 30), color='green')
            img2 = Image.new('RGB', (60, 30), color='blue')
        return img1, img2

    def test_get_image_hash(self):
        image_hasher = ImageHasher()

        duplicate_1, duplicate_2 = self._create_sample_images(duplicate=True)
        unique_1, unique_2 = self._create_sample_images(duplicate=False)

        hash1 = image_hasher.get_image_hash(duplicate_1)
        hash2 = image_hasher.get_image_hash(duplicate_2)
        assert hash1 == hash2

        hash1 = image_hasher.get_image_hash(unique_1)
        hash2 = image_hasher.get_image_hash(unique_2)
        assert hash1 != hash2

    def test_get_image_with_check(self):
        image_hasher = ImageHasher()

        duplicate_1, duplicate_2 = self._create_sample_images(duplicate=True)
        unique_1, unique_2 = self._create_sample_images(duplicate=False)

        assert image_hasher.get_image_if_is_not_duplicate(duplicate_1) == duplicate_1
        assert image_hasher.get_image_if_is_not_duplicate(duplicate_2) is None
        assert image_hasher.get_image_if_is_not_duplicate(unique_1) == unique_1
        assert image_hasher.get_image_if_is_not_duplicate(unique_2) == unique_2
