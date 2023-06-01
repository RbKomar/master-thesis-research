# src/main.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from data.dataset_generator import DatasetGenerator
from models.vgg_model import VGGModelTrainer
from models.resnet_model import ResNet152ModelTrainer
from models.googlenet_model import GoogLeNetModelTrainer
from models.xception_model import XceptionModelTrainer
from models.efficienet_model import EfficientNetModelTrainer
from models.baseline import BaselineTrainer

# Initialize model trainers
INPUT_SHAPE = (256, 256, 3)
EPOCHS = 10
vgg_trainer = VGGModelTrainer(input_shape=INPUT_SHAPE, epochs=EPOCHS)
vgg_trainer_no_imagenet = VGGModelTrainer(input_shape=INPUT_SHAPE, use_imagenet=False, epochs=EPOCHS)
resnet152_trainer = ResNet152ModelTrainer(input_shape=INPUT_SHAPE, epochs=EPOCHS)
resnet152_trainer_no_imagenet = ResNet152ModelTrainer(input_shape=INPUT_SHAPE, use_imagenet=False, epochs=EPOCHS)
xception_trainer = XceptionModelTrainer(input_shape=INPUT_SHAPE, epochs=EPOCHS)
xception_trainer_no_imagenet = XceptionModelTrainer(input_shape=INPUT_SHAPE, use_imagenet=False, epochs=EPOCHS)
efficientnet_trainer = EfficientNetModelTrainer(input_shape=INPUT_SHAPE, epochs=EPOCHS, save_model=False)
efficientnet_trainer_no_imagenet = EfficientNetModelTrainer(input_shape=INPUT_SHAPE, use_imagenet=False, epochs=EPOCHS, save_model=False)
googlenet_trainer = GoogLeNetModelTrainer(input_shape=INPUT_SHAPE, epochs=EPOCHS)
baseline = BaselineTrainer(input_shape=INPUT_SHAPE, epochs=EPOCHS)

# Initialize dataset generator
relative_path = os.path.join('..', '..', '..', 'data', 'master-thesis-data')
absolute_path = os.path.abspath(relative_path)
dataset_generator = DatasetGenerator(absolute_path, augment=False, batch_size=32)

models = [(vgg_trainer, f'VGG_{EPOCHS}_epochs'),
          (resnet152_trainer, f'ResNet152_{EPOCHS}_epochs'),
          (xception_trainer, f'Xception_{EPOCHS}_epochs'),
          (efficientnet_trainer, f'EfficientNet_{EPOCHS}_epochs')]

models_no_imagenet = [(vgg_trainer_no_imagenet, f'VGG_no_imagenet_{EPOCHS}_epochs'),
                      (resnet152_trainer_no_imagenet, f'ResNet152_no_imagenet_{EPOCHS}_epochs'),
                      (xception_trainer_no_imagenet, f'Xception_no_imagenet_{EPOCHS}_epochs'),
                      (efficientnet_trainer_no_imagenet, f'Xception_no_imagenet_{EPOCHS}_epochs')]

# Train models on different datasets
for dataset_name, dataset in dataset_generator.generate_datasets():
    for model_trainer, model_name in models:
        print(dataset_name, len(dataset.train), len(dataset.test), len(dataset.val))
        model_trainer.initialize_model()
        with tf.device('/GPU:0'):
            model_trainer.train(dataset.train, dataset.test, dataset_name, model_name)
        results = model_trainer.evaluate(dataset.val)
        dataset_generator.save_results(model_name, dataset_name, results, EPOCHS)
        tf.keras.backend.clear_session()

for dataset_name, dataset in dataset_generator.generate_datasets():
    for model_trainer, model_name in models_no_imagenet:
        print(dataset_name, len(dataset.train), len(dataset.test), len(dataset.val))
        model_trainer.initialize_model()
        with tf.device('/GPU:0'):
            model_trainer.train(dataset.train, dataset.test, dataset_name, model_name)
        results = model_trainer.evaluate(dataset.val)
        dataset_generator.save_results(model_name, dataset_name, results, EPOCHS)
        tf.keras.backend.clear_session()
    break

relative_path = os.path.join('..', '..', '..', 'data', 'master-thesis-data')
absolute_path = os.path.abspath(relative_path)
dataset_generator = DatasetGenerator(absolute_path, augment=False, obscure_percent=70,  batch_size=32)

# +
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_single_image(dataset_generator):
    dataset_name = None
    dataset = None

    for dataset_name, dataset in dataset_generator.generate_datasets():
        if dataset:
            break

    if not dataset:
        print("No dataset available.")
        return

    data_batch = next(iter(dataset))
    images, labels = data_batch

    # Select a random image from the batch
    index = np.random.randint(0, len(images))
    image = images[index]
    label = labels[index]

    # Convert the image tensor to a NumPy array
    image = image.numpy() if isinstance(image, tf.Tensor) else image

    # Plot the image
    plt.imshow(image)
    plt.title(f"Dataset: {dataset_name}\nLabel: {label}")
    plt.axis('off')
    plt.show()

plot_single_image(dataset_generator)

# -

# Train models on different datasets
for dataset_name, dataset in dataset_generator.generate_datasets():
    for model_trainer, model_name in models:
        print(dataset_name, len(dataset.train), len(dataset.test), len(dataset.val))
        model_trainer.initialize_model()
        with tf.device('/GPU:0'):
            model_trainer.train(dataset.train, dataset.test, dataset_name, model_name)
        results = model_trainer.evaluate(dataset.val)
        dataset_generator.save_results(model_name, dataset_name, results, EPOCHS)
        tf.keras.backend.clear_session()
    break

for dataset_name, dataset in dataset_generator.generate_datasets():
    for model_trainer, model_name in models_no_imagenet:
        print(dataset_name, len(dataset.train), len(dataset.test), len(dataset.val))
        model_trainer.initialize_model()
        with tf.device('/GPU:0'):
            model_trainer.train(dataset.train, dataset.test, dataset_name, model_name)
        results = model_trainer.evaluate(dataset.val)
        dataset_generator.save_results(model_name, dataset_name, results, EPOCHS)
        tf.keras.backend.clear_session()
    break

# ## GPU Memory Release

from numba import cuda

device = cuda.get_current_device()
device.reset()


# ## Testing batch size

# +
def test_dataset_batching(dataset, batch_size):
    for batch in dataset.take(1):  # Take one batch from the dataset
        data, labels = batch
        assert len(data) == batch_size, f"Data batch size is incorrect: expected {batch_size}, got {len(data)}"
        assert len(labels) == batch_size, f"Labels batch size is incorrect: expected {batch_size}, got {len(labels)}"
    print(f"Dataset batch size is correct: {batch_size}")


# You can use this function to test your datasets
for dataset_name, dataset in dataset_generator.generate_datasets():
    print(f"Testing dataset: {dataset_name}")
    test_dataset_batching(dataset.train, dataset_generator.batch_size)
    test_dataset_batching(dataset.val, dataset_generator.batch_size)
    test_dataset_batching(dataset.test, dataset_generator.batch_size)
