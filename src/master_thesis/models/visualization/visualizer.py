from matplotlib import pyplot as plt

from src.master_thesis.models.experiments.robustness.robustness_test import RobustnessTester


class ModelVisualizer:
    def __init__(self, model_results):
        self.model_results = model_results

    @staticmethod
    def plot_training_history(model_name, history):
        # This is a hypothetical method; replace with actual plotting logic
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Training History - {model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_model_comparison(comparison_results):
        # This is a hypothetical method; replace with actual comparison plotting logic
        plt.figure(figsize=(10, 5))
        for model_name, results in comparison_results.items():
            plt.plot(results, label=model_name)
        plt.title('Model Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


class RobustnessTesterVisualization(RobustnessTester):
    def generate_heatmaps(self, sample_images, last_conv_layer_name):
        """Generate Grad-CAM heatmaps for a sample of images."""
        heatmaps = []
        for image in sample_images:
            # Assuming binary classification, class_index is set to 1
            heatmap = VisualizationTools.grad_cam(self.model, image, class_index=1,
                                                  last_conv_layer_name=last_conv_layer_name)
            heatmaps.append(heatmap)
        return heatmaps

    def plot_heatmaps(self, heatmaps, sample_images):
        """Plot the Grad-CAM heatmaps overlayed on the sample images."""
        for heatmap, image in zip(heatmaps, sample_images):
            VisualizationTools.plot_grad_cam(heatmap, image)


class RobustnessTesterGradCAM(RobustnessTesterVisualization):
    def generate_sample_heatmaps(self, sample_images, model_type, last_conv_layer_name):
        """Generate Grad-CAM heatmaps for a sample of images from each dataset type."""
        datasets = ["original", "obscured", "incomplete"]
        all_heatmaps = {}

        for dataset_type in datasets:
            heatmaps = []
            if dataset_type == "original":
                test_data = self.generator.generate_combined_datasets()
            elif dataset_type == "obscured":
                test_data = self.generate_obscured_dataset()
            else:
                test_data = self.generate_incomplete_dataset()

            for image in sample_images:
                heatmap = VisualizationTools.grad_cam(model_type, image, class_index=1,
                                                      last_conv_layer_name=last_conv_layer_name)
                heatmaps.append(heatmap)

            all_heatmaps[dataset_type] = heatmaps

        return all_heatmaps

    def plot_all_heatmaps(self, all_heatmaps, sample_images):
        """Plot the Grad-CAM heatmaps overlayed on the sample images for each dataset type."""
        for dataset_type, heatmaps in all_heatmaps.items():
            for heatmap, image in zip(heatmaps, sample_images):
                title = f"Dataset: {dataset_type.capitalize()}"
                VisualizationTools.plot_grad_cam(heatmap, image, title=title)
