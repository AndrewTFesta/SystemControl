"""
@title
@description
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from itertools import combinations
from json import JSONDecodeError
from multiprocessing import Queue
from signal import signal, SIGINT

import matplotlib.pyplot as plt
from matplotlib import style

from SystemControl import DATA_DIR
from SystemControl.Classifiers.TF.TfClassifier import TrainParameters, TfClassifier
from SystemControl.utils.Misc import find_files_by_name


def train_model(train_params, verbosity, save_queue):
    tf_classifier = TfClassifier(train_params, force_overwrite=True, verbosity=verbosity)
    tf_classifier.train_and_evaluate()
    tf_classifier.display_eval_metrics()
    save_queue.put(tf_classifier.model_fname)
    return


def sort_windows(window_entry):
    window_len_str = window_entry[0]
    window_parts = window_len_str.split('-')
    window_val = 0
    for part in window_parts:
        window_val += int(part) * len(window_parts)
    return window_val


def sort_metric_files(metric_fname):
    model_dir = os.path.dirname(metric_fname)
    model_id = os.path.basename(model_dir)
    model_id_parts = model_id.split('_')

    model_window_str = str(model_id_parts[-3])
    model_window_parts = model_window_str.split('-')
    return len(model_window_parts), int(model_window_parts[0])


def plot_model_boxplot(data_dict: dict, data_source: str, fig_title: str, x_title: str, y_title: str,
                       fig_width: int = 12, fig_height: int = 12):
    style.use('ggplot')
    data_dict = sorted(data_dict.items(), key=sort_windows)
    x_labels = [entry[0] for entry in data_dict]
    y_vals = [entry[1] for entry in data_dict]
    fig_title = f'{fig_title}: {data_source}'

    fig, axes = plt.subplots(figsize=(fig_width, fig_height), facecolor='black')
    box_plot = axes.boxplot(y_vals, patch_artist=True)

    for box in box_plot['boxes']:
        box.set(color='xkcd:darkblue', linewidth=1)
        box.set(facecolor='xkcd:coral')

    for whisker in box_plot['whiskers']:
        whisker.set(color='xkcd:grey', linewidth=1)

    for cap in box_plot['caps']:
        cap.set(color='xkcd:indigo', linewidth=2)

    for median in box_plot['medians']:
        median.set(color='xkcd:crimson', linewidth=2)

    for flier in box_plot['fliers']:
        flier.set(marker='o', color='xkcd:orchid', alpha=0.2)

    axes.set_xticklabels(x_labels, rotation=90, size=18)
    axes.tick_params(axis='y', labelcolor='k', labelsize=18)

    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()

    axes.set_title(fig_title, size=24)
    axes.set_xlabel(x_title, size=24)
    axes.set_ylabel(y_title, size=24)

    for xtick in axes.get_xticklabels():
        xtick.set_color('k')
        text_str = xtick.get_text()
        if '20' in text_str:
            xtick.set_backgroundcolor('y')

    fig.tight_layout()

    save_dir = os.path.join(DATA_DIR, 'metric_plots', data_source)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_title = fig_title.replace(' ', '_')
    fig_title = fig_title.replace(':', '_')
    plot_fname = os.path.join(save_dir, f"{fig_title}.png")

    plt.savefig(plot_fname)
    plt.close()
    return


class ModelPool:

    def __init__(self, duration_list: list, train_parameters: TrainParameters):
        self.duration_list = duration_list
        self.train_parameters = train_parameters
        self.base_model_dir = os.path.join(DATA_DIR, 'models', 'cnn', train_parameters.data_source)
        self.model_dirs = self.build_model_metainfo()
        self.model_metrics = {}

        self._training_proc = None
        self._keep_training = True
        self._train_result_queue = Queue()
        return

    def build_model_metainfo(self):
        start_time = time.time()
        trained_model_names = find_files_by_name('trained_model', self.base_model_dir)
        model_info_list = []
        for model_name in trained_model_names:
            model_dir = os.path.dirname(model_name)
            model_id = os.path.basename(model_dir)
            model_ds = model_id.split('_')[0]

            if model_ds == self.train_parameters.data_source and self.__validate_model_dir(model_dir):
                model_info_list.append(model_dir)
        end_time = time.time()
        print(f'Time to find trained models: {end_time - start_time:0.4f}')
        print(f'Found {len(trained_model_names)} trained models')
        return model_info_list

    @staticmethod
    def __validate_model_dir(model_dir):
        req_files = {
            'epoch_metrics',
            'trained_model',
            'train_eval_metrics',
            'train_params'
        }
        mdl_files = {
            os.path.splitext(each_file)[0]
            for each_file in os.listdir(model_dir)
        }
        missing_files = req_files.difference(mdl_files)
        is_valid = len(missing_files) == 0
        return is_valid

    def build_model_train_eval_metrics(self):
        eval_dict = {}
        start_time = time.time()
        eval_file_list = sorted(find_files_by_name('train_eval_metrics', self.base_model_dir), key=sort_metric_files)
        for eval_file in eval_file_list:
            model_dir = os.path.dirname(eval_file)
            model_id = os.path.basename(model_dir)
            model_id_parts = model_id.split('_')
            model_ds = model_id_parts[0]
            model_subject = model_id_parts[1]
            model_window_str = str(model_id_parts[-3])

            if model_ds == self.train_parameters.data_source:
                try:
                    with open(eval_file, 'r+') as metric_file:
                        metric_data = json.load(metric_file)
                    if model_window_str not in eval_dict:
                        eval_dict[model_window_str] = {}
                    eval_dict[model_window_str][model_subject] = metric_data
                except JSONDecodeError as jde:
                    print(f'Failed to decode file: {eval_file}\n\t{jde}')
        end_time = time.time()
        print(f'Time to find trained models: {end_time - start_time:0.4f}')
        print(f'Found {len(eval_file_list)} trained models')
        self.model_metrics = eval_dict
        return eval_dict

    def plot_metrics(self):
        self.plot_train_time()
        self.plot_eval_time()
        self.plot_predict_time()
        self.plot_test_accuracy()
        return

    def plot_train_time(self):
        data = {}
        for window_len, window_entry in self.model_metrics.items():
            for subject_name, subject_entry in window_entry.items():
                train_time = subject_entry['train_time']
                if window_len not in data:
                    data[window_len] = []
                data[window_len].append(train_time)

        plot_model_boxplot(
            data_dict=data,
            data_source=self.train_parameters.data_source,
            fig_title='Train time vs Window lengths',
            x_title='Window lengths (s)',
            y_title='Train time (s)'
        )
        return

    def plot_eval_time(self):
        data = {}
        for window_len, window_entry in self.model_metrics.items():
            for subject_name, subject_entry in window_entry.items():
                eval_time = subject_entry['eval_time']
                if window_len not in data:
                    data[window_len] = []
                data[window_len].append(eval_time)

        plot_model_boxplot(
            data_dict=data,
            data_source=self.train_parameters.data_source,
            fig_title='Evaluation time vs Window lengths',
            x_title='Window lengths (s)',
            y_title='Evaluation time (s)'
        )
        return

    def plot_predict_time(self):
        data = {}
        for window_len, window_entry in self.model_metrics.items():
            for subject_name, subject_entry in window_entry.items():
                pred_time = subject_entry['predict_per_image_time']
                if window_len not in data:
                    data[window_len] = []
                data[window_len].append(pred_time)

        plot_model_boxplot(
            data_dict=data,
            data_source=self.train_parameters.data_source,
            fig_title='Prediction time per image vs Window lengths',
            x_title='Window lengths (s)',
            y_title='Prediction time per image (s)'
        )
        return

    def plot_test_accuracy(self):
        data = {}
        for window_len, window_entry in self.model_metrics.items():
            for subject_name, subject_entry in window_entry.items():
                test_acc = subject_entry['eval_metrics']['sparse_categorical_accuracy']
                if window_len not in data:
                    data[window_len] = []
                data[window_len].append(test_acc)

        plot_model_boxplot(
            data_dict=data,
            data_source=self.train_parameters.data_source,
            fig_title='Test accuracy vs Window lengths',
            x_title='Window lengths (s)',
            y_title='Test accuracy (%)'
        )
        return

    def get_model_ids(self):
        model_id_list = []
        for model_dir in self.model_dirs:
            model_id = os.path.basename(model_dir)
            model_id_list.append(model_id)
        return model_id_list

    def signal_handler(self, signal_received, frame):
        self._training_proc.terminate()
        self._keep_training = False
        return

    def train_missing_models(self, verbosity=0):
        signal(SIGINT, self.signal_handler)
        self._keep_training = True

        duration_combinations = []
        for chose_num in range(1, len(self.duration_list) + 1):
            duration_combinations.extend(list(combinations(self.duration_list, chose_num)))

        model_id_list = self.get_model_ids()
        heatmap_dir = os.path.join(DATA_DIR, 'heatmaps', f'data_source_{self.train_parameters.data_source}')
        subject_names = sorted([
            '_'.join(subject_dir.split('_')[1:])
            for subject_dir in os.listdir(heatmap_dir)
            if os.path.isdir(os.path.join(heatmap_dir, subject_dir))
        ])

        for each_subject in subject_names:
            for each_duration_list in duration_combinations:
                if not self._keep_training:
                    return
                each_train_param = self.train_parameters._replace(
                    chosen_being=each_subject,
                    window_lengths=each_duration_list
                )
                model_id = TfClassifier.build_model_id(each_train_param)
                if model_id not in model_id_list:
                    proc_args = (each_train_param, verbosity, self._train_result_queue)

                    self._training_proc = mp.Process(target=train_model, args=proc_args)
                    self._training_proc.start()
                    self._training_proc.join()
        return


def main(margs):
    ds_name = margs.get('data_source', 'Physio')
    target_column = margs.get('target', 'event')
    duration_list = ['0.20', '0.40', '0.60', '0.80', '1.00']
    interpolation_list = margs.get('interpolation', ['LINEAR', 'QUADRATIC', 'CUBIC'])
    num_epochs = margs.get('num_epochs', 20)
    batch_size = margs.get('batch_size', 16)
    learning_rate = margs.get('learning_rate', 1e-4)
    img_width = margs.get('img_width', 224)
    img_height = margs.get('img_height', 224)
    verbosity = margs.get('verbosity', 1)
    ###############################################################################
    train_params = TrainParameters(
        data_source=ds_name, chosen_being=None, window_lengths=None,
        interpolation_list=interpolation_list, target_column=target_column,
        img_width=img_width, img_height=img_height, learning_rate=learning_rate,
        batch_size=batch_size, num_epochs=num_epochs
    )

    model_pool = ModelPool(duration_list=duration_list, train_parameters=train_params)
    # model_pool.train_missing_models(verbosity=verbosity)
    model_pool.build_model_train_eval_metrics()
    model_pool.plot_metrics()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate multiple Tensorflow models.')
    parser.add_argument('--data_source', type=str, default=os.path.join('Physio'), choices=['Physio', 'recorded'],
                        help='name of the data source to use when training the model')
    parser.add_argument('--target', type=str, default='event',
                        help='target variable to be used as the class label')

    parser.add_argument('--duration', type=str, nargs='+', default=['0.20', '0.40', '0.60', '0.80', '1.00'],
                        choices=['0.20', '0.40', '0.60', '0.80', '1.00'],
                        help='list of window sizes to use when training the model')
    parser.add_argument('--interpolation', type=str, nargs='+', default=['LINEAR', 'QUADRATIC', 'CUBIC'],
                        choices=['LINEAR', 'QUADRATIC', 'CUBIC'],
                        help='list of window sizes to use when training the model')
    parser.add_argument('--img_width', type=int, default=224,
                        help='width to resize each image to before feeding to the model')
    parser.add_argument('--img_height', type=int, default=224,
                        help='height to resize each image to before feeding to the model')

    parser.add_argument('--num_epochs', type=int, default=200,
                        help='maximum number of epochs over which to train the model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='size of a training batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate to use to train the model')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='verbosity level to use when reporting model updates: 0 -> off, 1-> on')

    args = parser.parse_args()
    main(vars(args))
