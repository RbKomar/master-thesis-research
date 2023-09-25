import tensorflow as tf


class DataSetCreator:
    def __init__(self, img_paths, img_labels, image_parser, obscure_images_percent, batch_size):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.image_parser = image_parser
        self.obscure_images_percent = obscure_images_percent
        self.batch_size = batch_size

    def create_datasets(self):
        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = tf.data.Dataset.from_tensor_slices((self.img_paths[split], self.img_labels[split]))
            datasets[split] = datasets[split].map(
                lambda x, y: (self.image_parser(x, self.obscure_images_percent), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if split == "train":
                datasets[split] = datasets[split].shuffle(buffer_size=100)
            datasets[split] = datasets[split].batch(self.batch_size)
            datasets[split] = datasets[split].prefetch(tf.data.experimental.AUTOTUNE)
        return datasets
