import subprocess

def train_all_models():
    model_names = ['baseline', 'VGG16', 'ResNet']  # Add more models as needed
    dataset_names = ['ham10000', 'isic2016', 'isic2017', 'isic2019', 'isic2020']  # Add more datasets as needed

    for model_name in model_names:
        for dataset_name in dataset_names:
            subprocess.run(['python', 'src/models/train_model.py', model_name, dataset_name])

if __name__ == "__main__":
    train_all_models()
