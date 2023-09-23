import hashlib
from typing import Optional

import numpy as np
from PIL import Image


class ImageHasher:
    """Class to generate hashes for images and check if they are duplicates."""


    def __init__(self):
        self.existing_hashes = set()
        self.duplicates_counter = 0

    @staticmethod
    def get_image_hash(image: str | Image.Image):
        """Generate a hash for an image."""
        if isinstance(image, str):
            image = Image.open(image)
        image = np.array(image)
        image = image.flatten()
        image = image.tobytes()
        return hashlib.sha256(image).hexdigest()

    def get_image_if_is_not_duplicate(self, image: str | Image.Image) -> Optional[str | Image.Image]:
        """Generate a hash for an image and check if it already exists.
        If it does return None, otherwise return the image_path."""
        image_hash = self.get_image_hash(image)
        result = image  # default value
        if image_hash in self.existing_hashes:
            self.duplicates_counter += 1
            result = None
        else:
            self.existing_hashes.add(image_hash)
        return result

