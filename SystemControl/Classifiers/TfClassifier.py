"""
@title
@description
"""
import argparse
import collections
import json
import os
import random
import shutil
import sys
import time

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import style
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

from SystemControl import DATA_DIR
from utils.utilities import find_files_by_type, filter_list_of_dicts

TrainParameters = collections.namedtuple(
    'TrainParameters',
    'data_source, chosen_being, target_column, interpolation_list, window_lengths,'
    'img_height, img_width, learning_rate, batch_size, num_epochs'
)


def build_model_id(train_params: TrainParameters):
    interp_str = '-'.join(train_params.interpolation_list)
    window_str = '-'.join(train_params.window_lengths).replace('0.', '')
    model_str = f'{train_params.data_source}_{train_params.chosen_being}_{train_params.target_column}_' \
                f'{interp_str}_{window_str}_{train_params.img_width}w_{train_params.img_height}h'
    return model_str


def default_model_metrics():
    metrics_list = [
        keras.metrics.SparseCategoricalAccuracy(),
    ]
    return metrics_list


def default_model_callbacks(metric_fname, verbosity, save_checkpoints_file: str = None):
    fit_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', verbose=verbosity, patience=3, mode='auto', restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(filename=metric_fname, separator=',', append=True)
    ]
    if save_checkpoints_file:
        fit_callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=save_checkpoints_file, monitor='val_loss', verbose=verbosity, mode='auto',
            save_weights_only=False, save_best_only=True
        ))
    return fit_callbacks


def default_model_optimizer(lr):
    optimizer = keras.optimizers.Adam(lr=lr)
    return optimizer


def default_model_loss():
    loss_func = keras.losses.SparseCategoricalCrossentropy()
    return loss_func


def load_model(model_fname):
    model = None
    if os.path.isfile(model_fname):
        model = keras.models.load_model(model_fname)
    return model


def plot_confusion_matrix(y_true, y_pred, class_labels, title,
                          cmap: str = 'Blues', annotate_entries: bool = True, save_plot: str = None):
    style.use('ggplot')
    fig_title = f'Target: \'{title}\''
    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    lower_bound = np.min(y_true) - 0.5
    upper_bound = np.max(y_true) + 0.5

    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    xtick_marks = np.arange(conf_mat.shape[1])
    ytick_marks = np.arange(conf_mat.shape[0])

    ax.set_xticks(xtick_marks)
    ax.set_yticks(ytick_marks)

    ax.set_xbound(lower=lower_bound, upper=upper_bound)
    ax.set_ybound(lower=lower_bound, upper=upper_bound)
    ax.invert_yaxis()

    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

    ax.set_title(fig_title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    if annotate_entries:
        annot_format = '0.2f'
        thresh = conf_mat.max() / 2.
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                conf_entry = conf_mat[i, j]
                ax.text(
                    j, i, format(conf_entry, annot_format), ha='center', va='center',
                    color='white' if conf_entry > thresh else 'black'
                )
    fig.tight_layout()

    if save_plot:
        plt.savefig(save_plot)
    else:
        plt.show()
    plt.close()
    return


def split_data(data, targets, test_size=0.25, val_size=0.15):
    train_val_x, test_x, train_val_y, test_y = train_test_split(
        data, targets, test_size=test_size
    )

    train_x, val_x, train_y, val_y = train_test_split(
        train_val_x, train_val_y, test_size=val_size
    )

    data_dict = {
        'train': {'images': train_x, 'labels': train_y.to_numpy()},
        'validation': {'images': val_x, 'labels': val_y.to_numpy()},
        'test': {'images': test_x, 'labels': test_y.to_numpy()},
    }
    return data_dict


class TfClassifier:

    @property
    def model_type(self):
        return 'cnn'

    def __init__(self, train_params: TrainParameters, *,
                 save_checkpoints: bool = False, force_overwrite: bool = False, verbosity: int = 1):

        self._train_params = train_params
        self._verbosity = verbosity

        gpu_list = tf.config.experimental.list_physical_devices('GPU')
        if gpu_list:
            for gpu in gpu_list:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpu_list[-1], 'GPU')

        heatmap_dir = os.path.join(
            DATA_DIR, 'heatmaps',
            f'data_source_{self._train_params.data_source}',
            f'subject_{self._train_params.chosen_being}'
        )
        for duration in self._train_params.window_lengths:
            duration_data_dir = os.path.join(heatmap_dir, f'window_length_{duration}')
            if not os.path.isdir(duration_data_dir):
                raise ValueError(f'Dataset directory not found: {duration_data_dir}')

        self.dataset_directory = heatmap_dir
        self.model_id = build_model_id(train_params)
        self.model_dir = os.path.join(DATA_DIR, 'models', self.model_type, f'{self.model_id}')
        self.train_params_fname = os.path.join(self.model_dir, f'train_params.json')
        self.epoch_metrics_fname = os.path.join(self.model_dir, 'epoch_metrics.csv')
        self.triain_eval_metrics_fname = os.path.join(self.model_dir, f'train_eval_metrics.json')
        self.model_fname = os.path.join(self.model_dir, 'trained_model.hdf5')

        if os.path.isdir(self.model_dir) and force_overwrite:
            shutil.rmtree(self.model_dir)

        self._checkpoint_dir = None
        self._save_checkpoints_fname = None
        if save_checkpoints:
            self._checkpoint_dir = os.path.join(self.model_dir, f'model_checkpoints')
            self._save_checkpoints_fname = os.path.join(self._checkpoint_dir, f'val_loss_{{val_loss:0.4f}}.hdf5')
            if not os.path.isdir(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            if not os.path.isdir(self.model_dir):
                os.makedirs(self.model_dir)

        self._train_history = None
        self._train_time = -1
        self._eval_metrics = None
        self._eval_time = -1
        self._test_predictions = None
        self._predict_time = -1

        self._data_splits = None
        raw_data = self.__load_image_files()
        self.class_names = list(
            np.unique([each_entry[self._train_params.target_column] for each_entry in raw_data])
        )

        self.model = load_model(self.model_fname)
        if self.model:
            if self._verbosity > 0:
                print('prev model found')

            # todo handle case where loading info fails for some reason (reset model?)
            self.__load_epoch_history()
            self.__load_eval_metrics()
        else:
            if self._verbosity > 0:
                print('no prev model found')
            filtered_x, filtered_y = self.__filter_data_entries(raw_data)
            self._data_splits = split_data(filtered_x, filtered_y)
            self.model = self.__build_new_model()
        return

    def train_and_evaluate(self):
        self._train_history, self._train_time = self.train(
            self._data_splits['train']['images'], self._data_splits['train']['labels'],
            self._data_splits['validation']['images'], self._data_splits['validation']['labels'],
        )
        self._eval_metrics, self._eval_time = self.evaluate(
            self._data_splits['test']['images'], self._data_splits['test']['labels']
        )
        self._test_predictions, self._predict_time = self.predict(self._data_splits['test']['images'])

        # todo save data splits
        self.__save_train_parameters()
        self.__save_train_eval_predict_metrics()
        self.__save_model()

        self.__plot_train_metrics()
        self.__plot_test_metrics()
        self.__plot_model()
        return True

    def __build_new_model(self):
        input_shape = (self._train_params.img_height, self._train_params.img_width, 3)
        num_classes = len(self.class_names)
        model = keras.Sequential()
        # base architecture
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

        # dense layers
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(num_classes, activation='sigmoid'))

        model.compile(
            optimizer=default_model_optimizer(self._train_params.learning_rate),
            loss=default_model_loss(),
            metrics=default_model_metrics()
        )
        return model

    def __filter_data_entries(self, data_to_filter):
        filter_criteria = {
            'subject': [self._train_params.chosen_being],
            'interpolation': self._train_params.interpolation_list,
            'data_source': [self._train_params.data_source],
            'window_length': self._train_params.window_lengths
        }
        filtered_dataset = filter_list_of_dicts(data_to_filter, filter_criteria)
        rand_generator = random.Random()
        rand_generator.shuffle(filtered_dataset)
        filtered_df = pd.DataFrame(filtered_dataset)

        filtered_x = filtered_df["path"]
        filtered_y = filtered_df[self._train_params.target_column]

        target_str_to_idx = {class_name: class_idx for class_idx, class_name in enumerate(self.class_names)}
        filtered_y.replace(target_str_to_idx, inplace=True)
        return filtered_x, filtered_y

    def __load_image_files(self):
        img_files = find_files_by_type('png', self.dataset_directory)
        data_list = []

        pbar = None
        if self._verbosity > 0:
            pbar = tqdm(img_files, desc='Reading dataset directory hierarchy', file=sys.stdout)

        for each_file in img_files:
            file_parts = each_file.split(os.path.sep)
            param_dict = {
                '_'.join(part.split('_')[:-1]): part.split('_')[-1]
                for part in file_parts
                if len(part.split('_')) > 1
            }
            param_dict["path"] = each_file
            data_list.append(param_dict)
            if self._verbosity > 0:
                pbar.update(1)
        if self._verbosity > 0:
            pbar.close()
        return data_list

    def __load_images(self, img_paths):
        pbar = None
        if self._verbosity > 0:
            pbar = tqdm(img_paths, desc=f'Loading images', file=sys.stdout)

        img_list = []
        for each_path in img_paths:
            img = cv2.imread(each_path)
            img_width = self._train_params.img_width if self._train_params.img_width > 0 else img.shape(1)
            img_height = self._train_params.img_height if self._train_params.img_height > 0 else img.shape(0)
            img_dims = (img_width, img_height)
            img = cv2.resize(img, img_dims)
            img_list.append(img)
            if self._verbosity > 0:
                pbar.update(1)
        if self._verbosity > 0:
            pbar.close()
        np_images = np.asarray(img_list)
        np_images = np_images / 255.0
        return np_images

    def __load_epoch_history(self):
        with open(self.epoch_metrics_fname, 'r+') as epoch_history_file:
            epoch_history = pd.read_csv(epoch_history_file)
        return epoch_history

    def __load_eval_metrics(self):
        with open(self.triain_eval_metrics_fname, 'r+') as metrics_file:
            self._eval_metrics = json.load(metrics_file)
        return

    def __save_train_eval_predict_metrics(self):
        save_dict = {
            'eval_metrics': self._eval_metrics,
            'eval_time': self._eval_time,
            'train_time': self._train_time,
            'predict_per_image_time': self._predict_time
        }

        with open(self.triain_eval_metrics_fname, 'w+') as metrics_file:
            json.dump(save_dict, metrics_file, indent=2)
        return

    def __save_train_parameters(self):
        with open(self.train_params_fname, 'w+') as meta_file:
            json.dump(self._train_params._asdict(), meta_file, indent=2)
        return

    def __save_model(self):
        self.model.save(self.model_fname)
        return

    def train(self, train_image_paths, train_labels, val_image_paths, val_labels):
        if self._verbosity > 0:
            print(f'Starting training: {self.model_id}')
            print(f'\tdata source:       {self._train_params.data_source}')
            print(f'\tchosen_being:      {self._train_params.chosen_being}')
            print(f'\ttarget_column:     {self._train_params.target_column}')
            print(f'\tinterpolation:     {self._train_params.interpolation_list}')
            print(f'\twindow size:       {self._train_params.window_lengths}')
            print(f'\tlearning_rate:     {self._train_params.learning_rate:0.6f}')
            print(f'\tbatch_size:        {self._train_params.batch_size}')
            print(f'\tnum_epochs:        {self._train_params.num_epochs}')

        loaded_train_images = self.__load_images(train_image_paths)
        loaded_val_images = self.__load_images(val_image_paths)
        start_time = time.time()
        self.model.fit(
            loaded_train_images, train_labels,
            epochs=self._train_params.num_epochs, verbose=self._verbosity, batch_size=self._train_params.batch_size,
            validation_data=(loaded_val_images, val_labels),
            callbacks=default_model_callbacks(
                metric_fname=self.epoch_metrics_fname,
                verbosity=self._verbosity,
                save_checkpoints_file=self._save_checkpoints_fname
            ),
        )
        end_time = time.time()
        delta_time = end_time - start_time
        train_time = delta_time

        train_history = self.__load_epoch_history()
        return train_history, train_time

    def evaluate(self, eval_images, eval_labels):
        if self._verbosity > 0:
            print(f'Evaluating model: {self.model_id}')
            print(f'Number of test images: {len(self._data_splits["test"]["images"])}')

        loaded_images = self.__load_images(eval_images)
        start_time = time.time()
        model_performance = self.model.evaluate(
            loaded_images, eval_labels, verbose=0
        )

        eval_metrics = {
            metric_name: float(model_performance[metric_index])
            for metric_index, metric_name in enumerate(self.model.metrics_names)
        }
        end_time = time.time()
        delta_time = end_time - start_time
        eval_time = delta_time
        return eval_metrics, eval_time

    def predict(self, image_paths):
        if self._verbosity > 0:
            print(f'Predicting using model: {self.model_id}')
            print(f'Number of predictions: {len(image_paths)}')

        loaded_images = None
        if isinstance(image_paths, pd.Series):
            if isinstance(image_paths.iloc[-1], str):
                loaded_images = self.__load_images(image_paths)
        else:
            if isinstance(image_paths[-1], str):
                loaded_images = self.__load_images(image_paths)
            else:
                raise TypeError(f'Unrecognized type for image paths: {type(image_paths)}')

        start_time = time.time()
        model_predictions = self.model.predict(loaded_images)
        end_time = time.time()
        delta_time = end_time - start_time
        predict_time = delta_time / len(loaded_images)
        return model_predictions, predict_time

    def display_dataset(self):
        train_event_counts = pd.Series(self._data_splits['train']['labels']).value_counts()
        val_event_counts = pd.Series(self._data_splits['validation']['labels']).value_counts()
        test_event_counts = pd.Series(self._data_splits['test']['labels']).value_counts()

        train_img_shape = self._data_splits['train']['images'].shape
        val_img_shape = self._data_splits['validation']['images'].shape
        test_img_shape = self._data_splits['test']['images'].shape

        num_train_images = len(self._data_splits['train']['images'])
        num_val_images = len(self._data_splits['validation']['images'])
        num_test_images = len(self._data_splits['test']['images'])
        total_count = num_train_images + num_val_images + num_test_images

        num_train_labels = len(self._data_splits['train']['labels'])
        num_val_labels = len(self._data_splits['validation']['labels'])
        num_test_labels = len(self._data_splits['test']['labels'])

        print('=====================================================================')
        print(f'Data source: {self._train_params.data_source}')
        print(f'Chosen being: {self._train_params.chosen_being}')
        print(f'Number of interpolation types: {len(self._train_params.interpolation_list)}')
        print(f'\t{", ".join(self._train_params.interpolation_list)}')
        print(f'Number of classes: {len(self.class_names)}')
        print(f'\t{", ".join(self.class_names)}')
        print(f'Number of entries in dataset: {total_count}')
        print('=====================================================================')
        print(f'Number of train images: {num_train_images}')
        for event_idx, event_count in train_event_counts.iteritems():
            print(f'\t{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / num_train_labels:0.4f} %)')
        print()
        print('=====================================================================')
        print(f'Number of validation images: {num_val_images}')
        for event_idx, event_count in val_event_counts.iteritems():
            print(f'\t{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / num_val_labels:0.4f} %)')
        print()
        print('=====================================================================')
        print(f'Number of test images: {num_test_images}')
        for event_idx, event_count in test_event_counts.iteritems():
            print(f'\t{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / num_test_labels:0.4f} %)')
        print()
        print('=====================================================================')
        print(f'Shape of train images:      {train_img_shape}')
        print(f'Shape of validation images: {val_img_shape}')
        print(f'Shape of test images:       {test_img_shape}')
        print('=====================================================================')
        return

    def display_samples(self):
        train_images = self._data_splits['train']['images']
        val_images = self._data_splits['validation']['images']
        test_images = self._data_splits['test']['images']

        train_labels = self._data_splits['train']['labels']
        val_labels = self._data_splits['validation']['labels']
        test_labels = self._data_splits['test']['labels']

        self.__show_samples(train_images, train_labels, title='Train', num_rows=5, num_cols=5, save_plot=True)
        self.__show_samples(val_images, val_labels, title='Validation', num_rows=5, num_cols=5, save_plot=True)
        self.__show_samples(test_images, test_labels, title='Test', num_rows=5, num_cols=5, save_plot=True)
        return

    def display_train_history(self):
        print('==========================================================================')
        print('Training history')
        sep = '\t'
        for metric_name, val_list in self._train_history.items():
            print(f'{sep}{metric_name}: ', end='')
            sep = ''
            for metric_val in val_list:
                print(f'{sep}{metric_val:0.4f}', end='')
                sep = ', '
            sep = '\n\t'
        print()
        print('==========================================================================')
        return

    def display_eval_metrics(self):
        print('==========================================================================')
        print('Evaluation metrics')
        sep = '\t'
        for metric_name, metric_val in self._eval_metrics.items():
            print(f'{sep}{metric_name}: {metric_val:0.4f}', end='')
            sep = ', '
        print()
        print('==========================================================================')
        return

    def display_model(self):
        print('==========================================================================')
        print('Model summary')
        self.model.summary()
        print('==========================================================================')
        return

    def __show_samples(self, sample_imgs, sample_labels, title: str,
                       num_rows: int = 5, num_cols: int = 5, save_plot: bool = False):
        style.use('ggplot')
        fig, ax_list = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
        fig.suptitle(f'{title} images: top {num_rows * num_cols}')
        for each_ax_idx, each_ax in enumerate(ax_list.flat):
            each_ax.set_xticks([])
            each_ax.set_yticks([])
            each_ax.imshow(sample_imgs[each_ax_idx])
            each_ax.set_xlabel(sample_labels[each_ax_idx])

        if save_plot:
            plt.savefig(os.path.join(
                self.model_dir, f'samples_{title}_rows_{num_rows}_cols_{num_cols}.png')
            )
        else:
            plt.show()
        plt.close()
        return

    def __plot_train_metrics(self):
        for metric_name in self._train_history:
            if not metric_name.startswith('epoch') and not metric_name.startswith('val_'):
                self.__plot_history_metric(metric_name, save_plot=True)
        return

    def __plot_test_metrics(self):
        top_test_preds = [np.argmax(each_pred) for each_pred in self._test_predictions]
        annotate_conf = len(self.class_names) < 20
        save_fname = os.path.join(
            self.model_dir, f'confusion_matrix_{self._train_params.target_column}.png'
        )
        plot_confusion_matrix(
            y_pred=top_test_preds, y_true=self._data_splits['test']['labels'], class_labels=self.class_names,
            title=self._train_params.target_column, annotate_entries=annotate_conf, save_plot=save_fname
        )
        return

    def __plot_history_metric(self, metric_name, save_plot: bool = False):
        style.use('ggplot')
        fig, ax = plt.subplots()

        train_vals = self._train_history[f'{metric_name}']
        validation_vals = self._train_history[f'val_{metric_name}']
        epoch_list = np.array(range(1, len(train_vals) + 1))

        ax.plot(epoch_list, train_vals, color='red', label=metric_name)
        ax.plot(epoch_list, validation_vals, color='black', label=f'validation {metric_name}')

        upper_y = np.max((train_vals, validation_vals)) * 1.2
        ax.set_ylim([0, upper_y])

        ax.set_xlim([epoch_list[0] - 0.5, epoch_list[-1] + 0.5])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)

        ax.legend(loc='best')
        ax.set_title(f'Model history: \'{metric_name}\'')

        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='cyan')
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='blue')
        ax.tick_params(which='both', top=False, left=False, right=False, bottom=False)

        if save_plot:
            plt.savefig(os.path.join(self.model_dir, f'metric_plot_{metric_name}.png'))
        else:
            plt.show()
        plt.close()
        return

    def __plot_model(self):
        """
        pip install pydot
        pip install pydotplus
        pip install graphviz
            https://www.graphviz.org/download/
            Make sure that the directory containing the dot executable is on your system's path
        """
        model_fname = os.path.join(self.model_dir, f'model_architecture.png')
        keras.utils.plot_model(self.model, model_fname, show_shapes=True)
        return


def main(margs):
    display_dataset = False
    display_samples = False
    display_metrics = True
    display_train_history = True
    display_model = False
    #############################################
    ds_name = margs.get('data_source', 'Physio')
    target_column = margs.get('target', 'event')
    subject_name = margs.get('subject_name', 'S001')
    duration_list = margs.get('duration', ['0.20'])
    interpolation_list = margs.get('interpolation', ['LINEAR', 'QUADRATIC', 'CUBIC'])
    force_overwrite = margs.get('force_overwrite', False)
    #############################################
    num_epochs = margs.get('num_epochs', 20)
    batch_size = margs.get('batch_size', 16)
    learning_rate = margs.get('learning_rate', 1e-4)
    verbosity = margs.get('verbosity', 1)
    img_width = margs.get('img_width', 224)
    img_height = margs.get('img_height', 224)
    #############################################
    train_params = TrainParameters(
        data_source=ds_name, chosen_being=subject_name, target_column=target_column,
        interpolation_list=interpolation_list, window_lengths=duration_list,
        img_width=img_width, img_height=img_height, learning_rate=learning_rate,
        batch_size=batch_size, num_epochs=num_epochs
    )

    tf_classifier = TfClassifier(
        train_params,
        force_overwrite=force_overwrite,
        verbosity=verbosity,
    )

    tf_classifier.train_and_evaluate()

    if display_dataset:
        tf_classifier.display_dataset()
    if display_samples:
        tf_classifier.display_samples()
    if display_train_history:
        tf_classifier.display_train_history()
    if display_metrics:
        tf_classifier.display_eval_metrics()
    if display_model:
        tf_classifier.display_model()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a Tensorflow classifier.')
    parser.add_argument('--data_source', type=str, default=os.path.join('Physio'), choices=['Physio', 'recorded'],
                        help='name of the data source to use when training the model')
    parser.add_argument('--target', type=str, default='event',
                        help='target variable to be used as the class label')
    parser.add_argument('--subject_name', type=str, default='S001',
                        help='Name of subject to use when training this classifier')
    parser.add_argument('--force_overwrite', action='store_true',
                        help='Overwrites any previously trained model that was trained using the specified parameters')

    parser.add_argument('--duration', type=str, nargs='+', default=['0.20'], choices=['0.20', '0.40', '0.60'],
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
