"""
@title
@description
"""
import argparse
import multiprocessing as mp
import os
import time
from itertools import combinations
from multiprocessing import Queue

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from SystemControl import DATA_DIR
from SystemControl.Classifiers.TfClassifier import TrainParameters, TfClassifier, build_model_id
from SystemControl.utilities import find_files_by_name


def train_model(train_params, verbosity, save_queue):
    tf_classifier = TfClassifier(train_params, force_overwrite=True, verbosity=verbosity)
    tf_classifier.train_and_evaluate()
    save_queue.put(tf_classifier.model_fname)
    return


class ModelPool:

    def __init__(self, data_source_list: list, duration_list: list, train_parameters: TrainParameters):
        self.data_source_list = data_source_list
        self.duration_list = duration_list
        self.train_parameters = train_parameters
        self.base_model_dir = os.path.join(DATA_DIR, 'models')
        self.model_dirs = self.build_model_metainfo()
        return

    def build_model_metainfo(self):
        start_time = time.time()
        trained_model_names = find_files_by_name('trained_model', self.base_model_dir)
        model_info_list = []
        for model_name in trained_model_names:
            model_dir = os.path.dirname(model_name)
            model_info_list.append(model_dir)
        end_time = time.time()
        print(f'Time to find trained models: {end_time - start_time:0.4f}')
        print(f'Found {len(trained_model_names)} trained models')
        return model_info_list

    def __validate_model_dir(self, model_dir):
        return

    def plot_metrics(self):
        return

    def save_metrics_csv(self):
        return

    def get_model_ids(self):
        model_id_list = []
        for model_dir in self.model_dirs:
            model_id = os.path.basename(model_dir)
            model_id_list.append(model_id)
        return model_id_list

    def train_models(self, verbosity=0):
        duration_combinations = []
        for chose_num in range(1, len(self.duration_list) + 1):
            duration_combinations.extend(list(combinations(self.duration_list, chose_num)))

        model_id_list = self.get_model_ids()
        train_result_queue = Queue()
        train_time_list = []
        for data_source in self.data_source_list:
            heatmap_dir = os.path.join(DATA_DIR, 'heatmaps', f'data_source_{data_source}')
            subject_names = [
                subject_dir.split('_')[-1]
                for subject_dir in os.listdir(heatmap_dir)
                if os.path.isdir(os.path.join(heatmap_dir, subject_dir))
            ]
            print(f'Starting training of {len(subject_names)} models')
            for each_subject in subject_names:
                for each_duration_list in duration_combinations:
                    each_train_param = self.train_parameters._replace(
                        data_source=data_source, chosen_being=each_subject, window_lengths=each_duration_list
                    )
                    model_id = build_model_id(each_train_param)
                    if model_id not in model_id_list:
                        start_time = time.time()
                        proc_args = (each_train_param, verbosity, train_result_queue)
                        train_proc = mp.Process(target=train_model, args=proc_args)
                        train_proc.start()
                        train_proc.join()
                        end_time = time.time()
                        delta_time = end_time - start_time
                        train_time_list.append(delta_time)
                        print(f'Time to train model: {delta_time:0.4f}')
                    else:
                        print(f'Model already trained:\n\t{model_id}')
        return train_result_queue, train_time_list


def main(margs):
    ds_name = margs.get('data_source', 'Physio')
    target_column = margs.get('target', 'event')
    duration_list = ['0.20', '0.40', '0.60']
    interpolation_list = margs.get('interpolation', ['LINEAR', 'QUADRATIC', 'CUBIC'])
    num_epochs = margs.get('num_epochs', 20)
    batch_size = margs.get('batch_size', 16)
    learning_rate = margs.get('learning_rate', 1e-4)
    img_width = margs.get('img_width', 224)
    img_height = margs.get('img_height', 224)
    verbosity = margs.get('verbosity', 0)
    train_params = TrainParameters(
        data_source=None, chosen_being=None, window_lengths=None,
        interpolation_list=interpolation_list, target_column=target_column,
        img_width=img_width, img_height=img_height, learning_rate=learning_rate,
        batch_size=batch_size, num_epochs=num_epochs
    )

    model_pool = ModelPool(data_source_list=[ds_name], duration_list=duration_list, train_parameters=train_params)
    model_pool.train_models(verbosity=verbosity)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate multiple Tensorflow models.')
    parser.add_argument('--data_source', type=str, default=os.path.join('Physio'), choices=['Physio', 'recorded'],
                        help='name of the data source to use when training the model')
    parser.add_argument('--target', type=str, default='event',
                        help='target variable to be used as the class label')

    parser.add_argument('--duration', type=str, nargs='+', default=['0.20', '0.40', '0.60'],
                        choices=['0.20', '0.40', '0.60'],
                        help='list of window sizes to use when training the model')
    parser.add_argument('--interpolation', type=str, nargs='+', default=['LINEAR', 'QUADRATIC', 'CUBIC'],
                        choices=['LINEAR', 'QUADRATIC', 'CUBIC'],
                        help='list of window sizes to use when training the model')
    parser.add_argument('--img_width', type=int, default=224,
                        help='width to resize each image to before feeding to the model')
    parser.add_argument('--img_height', type=int, default=224,
                        help='height to resize each image to before feeding to the model')

    parser.add_argument('--num_epochs', type=int, default=20,
                        help='maximum number of epochs over which to train the model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='size of a training batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate to use to train the model')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='verbosity level to use when reporting model updates: 0 -> off, 1-> on')

    args = parser.parse_args()
    main(vars(args))
