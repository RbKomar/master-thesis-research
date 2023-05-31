import time
import pandas as pd
from baseline import create_model as create_baseline_model, compile_model as compile_baseline_model

from src.data.make_dataset import load_data


def train_and_compare(models, model_names, train_dataset, val_dataset, epochs=10):
    histories = []
    training_times = []
    for model, model_name in zip(models, model_names):
        print(f'Training {model_name}...')
        start_time = time.time()
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        end_time = time.time()
        training_time = end_time - start_time
        histories.append(history)
        training_times.append(training_time)
        model.save(f'models/{model_name}.h5')
        print(f'Training time for {model_name}: {training_time} seconds.')
    return histories, training_times


if __name__ == '__main__':
    data_dir = 'path/to/data'
    train_dataset, val_dataset, test_dataset = load_data(data_dir)

    baseline_model = create_baseline_model()
    compile_baseline_model(baseline_model)

    models = [baseline_model, ...]  # lista modeli
    model_names = ['baseline', ...]  # lista nazw modeli

    histories, training_times = train_and_compare(models, model_names, train_dataset, val_dataset, epochs=10)

    training_times_df = pd.DataFrame({
        'model': model_names,
        'training_time': training_times
    })
    training_times_df.to_csv('training_times.csv', index=False)

    for history, model_name in zip(histories, model_names):
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f'{model_name}_history.csv', index=False)
