from src.master_thesis.data.handler.data_handler import DataHandler
from src.master_thesis.data.loader.base_loader import BaseLoader
from src.master_thesis.data.processing.image_processor import ImageParser


class ISICLoader(BaseLoader):
    @staticmethod
    def load_isic_data(data_path, batch_size=32, obscure_images_percent=0.0, year=2016, image_parser=ImageParser):
        data_handler = DataHandler(data_path, ".jpg", batch_size, image_parser.parse_image, obscure_images_percent,
                                   train_prop=0.7, val_prop=0.15, year=year)
        return data_handler.get_datasets()

    def load(self, data_path, batch_size=32, obscure_images_percent=0.0):
        year = self._get_year(data_path)
        if year and 2016 <= year < 2021:
            return self.load_isic_data(data_path, batch_size, obscure_images_percent=obscure_images_percent, year=year)
        else:
            raise ValueError(f"Invalid dataset name for: {data_path}")

    @staticmethod
    def _get_year(data_path):
        if "ham10000" in data_path.lower():
            return 2018
        if "isic" in data_path.lower():
            try:
                return int(data_path[-4:])
            except ValueError:
                return None
        return None


