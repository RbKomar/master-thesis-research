class DatasetLoader:
    @staticmethod
    def load_isic_data(data_path, batch_size, obscure_images_percent=0.0, year=2016):
        return parse_data(data_path, ".jpg", batch_size, obscure_images_percent=obscure_images_percent, year=year)

    def load_dataset(self, data_path, batch_size, obscure_images_percent=0.0):
        result = None
        if "isic2016" in data_path.lower():
            result = self.load_isic_data(data_path, batch_size, obscure_images_percent=obscure_images_percent, year=2016)
        elif "isic2017" in data_path.lower():
            result = self.load_isic_data(data_path, batch_size, obscure_images_percent=obscure_images_percent, year=2017)
        elif "isic2018" in data_path.lower():
            result = self.load_isic_data(data_path, batch_size, obscure_images_percent=obscure_images_percent, year=2018)
        elif "isic2019" in data_path.lower():
            result = self.load_isic_data(data_path, batch_size, obscure_images_percent=obscure_images_percent, year=2019)
        elif "isic2020" in data_path.lower():
            result = self.load_isic_data(data_path, batch_size, obscure_images_percent=obscure_images_percent, year=2020)
        else:
            raise ValueError(f"Invalid dataset name for: {data_path}")

        return result
