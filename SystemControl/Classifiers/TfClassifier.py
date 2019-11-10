"""
@title
@description
"""
import argparse
import collections
import json
import os
import random
import time

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

"""
have to set environment variable before importing tensorflow

0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataTransformer import Interpolation
from SystemControl.utilities import find_files_by_type, filter_list_of_dicts

TrainParameters = collections.namedtuple(
    'TrainParameter',
    [
        'source_list', 'chosen_beings', 'target_column', 'interpolation_list',
        'start_padding', 'end_padding', 'img_dims', 'learning_rate', 'batch_size', 'num_epochs'
    ]
)


class TfClassifier:

    def __init__(self, dataset_directory, num_subjects: int, target: str,
                 interpolation_types: list, start_padding: list, end_padding: list, model_id: str,
                 *, verbosity: int = 1, learning_rate: float = 1e-4, num_epochs: int = 20, batch_size: int = 1,
                 resize_height: int = 224, resize_width: int = 224, rand_seed: int = 42):
        time_stamp = time.strftime('%y_%m_%d_%H_%M_%S', time.gmtime())
        self.model_name = f'model_{model_id}'
        self.model_dir = os.path.join(DATA_DIR, 'models', 'tf', f'{self.type_name}', f'{self.model_name}_{time_stamp}')
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        self._rand_seed = rand_seed
        self._verbosity = verbosity

        self._data_directory = dataset_directory
        raw_data = self.__load_image_files()

        self.class_names: list = list(np.unique([each_entry[target] for each_entry in raw_data]))
        source_list = list(np.unique([each_entry['source'] for each_entry in raw_data]))
        potential_subjects = list(np.unique([each_entry['subject'] for each_entry in raw_data]))

        rand_generator = random.Random(self._rand_seed)
        chosen_beings = sorted(rand_generator.sample(potential_subjects, k=num_subjects))

        self._train_params = TrainParameters(
            source_list=source_list, chosen_beings=chosen_beings, target_column=target,
            interpolation_list=interpolation_types, start_padding=start_padding, end_padding=end_padding,
            img_dims=[resize_width, resize_height], learning_rate=learning_rate,
            batch_size=batch_size, num_epochs=num_epochs
        )
        self.save_train_params()

        filtered_dataframe = self.__filter_data_entries(raw_data)
        self._data_splits = self.__split_data(filtered_dataframe)

        self.model = self.__build_cnn_v0()

        self._train_history = None
        self._eval_metrics = None
        return

    @property
    def type_name(self):
        return 'cnn'

    def __load_image_files(self):
        img_files = find_files_by_type('png', self._data_directory)
        data_list = []
        for each_file in tqdm(img_files, desc='Reading dataset directory hierarchy'):
            file_parts = each_file.split(os.path.sep)
            file_id, file_ext = os.path.splitext(file_parts[-1])
            event_type = file_parts[-2]
            file_subject = file_parts[-3]
            file_interp = file_parts[-4]
            file_epad = file_parts[-5]
            file_spad = file_parts[-6]
            file_source = file_parts[-7]
            data_entry = {
                'event': event_type, 'subject': file_subject, 'interpolation': file_interp, 'start_padding': file_spad,
                'end_padding': file_epad, 'source': file_source, 'id': file_id, 'path': each_file
            }
            data_list.append(data_entry)
        return data_list

    def __filter_data_entries(self, data_to_filter):
        filter_criteria = {
            'subject': self._train_params.chosen_beings,
            'interpolation': self._train_params.interpolation_list,
            'source': self._train_params.source_list,
            'end_padding': self._train_params.end_padding
        }
        filtered_dataset = filter_list_of_dicts(data_to_filter, filter_criteria)
        rand_generator = random.Random(self._rand_seed)
        rand_generator.shuffle(filtered_dataset)
        filtered_df = pd.DataFrame(filtered_dataset)
        target_str_to_idx = {class_name: class_idx for class_idx, class_name in enumerate(self.class_names)}
        filtered_df.replace(target_str_to_idx, inplace=True)
        return filtered_df

    def __build_cnn_v0(self):
        input_shape = (self._train_params.img_dims[1], self._train_params.img_dims[0], 3)
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
            optimizer=self.default_model_optimizer(self._train_params.learning_rate),
            loss=self.default_model_loss(),
            metrics=self.default_model_metrics()
        )
        return model

    def default_model_callbacks(self):
        fit_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', verbose=self._verbosity, patience=3, mode='auto', restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'epoch_{epoch:02d}-val_loss_{val_loss:.2f}.hdf5'),
                monitor='val_loss', verbose=self._verbosity, mode='auto', save_weights_only=False, save_best_only=True
            ),
            keras.callbacks.CSVLogger(
                filename=os.path.join(self.model_dir, 'epoch_metrics.csv'),
                separator=',', append=True
            )
        ]
        return fit_callbacks

    @staticmethod
    def default_model_metrics():
        metrics_list = [
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.CategoricalHinge(),
        ]
        return metrics_list

    @staticmethod
    def default_model_optimizer(lr):
        optimizer = keras.optimizers.Adam(lr=lr)
        return optimizer

    @staticmethod
    def default_model_loss():
        loss_func = keras.losses.SparseCategoricalCrossentropy()
        return loss_func

    def __split_data(self, data_to_split, test_size=0.25, val_size=0.15):
        train_val_df, test_filtered_df = train_test_split(
            data_to_split, test_size=test_size, random_state=self._rand_seed
        )
        train_filtered_df, val_filtered_df = train_test_split(
            train_val_df, test_size=val_size, random_state=self._rand_seed
        )

        data_dict = {
            'train': {
                'images': self.__load_images(train_filtered_df['path']),
                'labels': train_filtered_df[self._train_params.target_column].to_numpy()
            },
            'validation': {
                'images': self.__load_images(val_filtered_df['path']),
                'labels': val_filtered_df[self._train_params.target_column].to_numpy()
            },
            'test': {
                'images': self.__load_images(test_filtered_df['path']),
                'labels': test_filtered_df[self._train_params.target_column].to_numpy()
            },
        }
        return data_dict

    def __load_images(self, img_paths):
        img_list = []
        for each_path in tqdm(img_paths, desc=f'Loading images'):
            img = cv2.imread(each_path)
            img = cv2.resize(img, (self._train_params.img_dims[0], self._train_params.img_dims[1]))
            img_list.append(img)
        np_images = np.asarray(img_list)
        np_images = np_images / 255.0
        return np_images

    def train(self):
        self._train_history = self.model.fit(
            self._data_splits['train']['images'], self._data_splits['train']['labels'],
            epochs=self._train_params.num_epochs, verbose=self._verbosity, batch_size=self._train_params.batch_size,
            callbacks=self.default_model_callbacks(),
            validation_data=(self._data_splits['validation']['images'], self._data_splits['validation']['labels'])
        )
        return self

    def evaluate(self):
        self._eval_metrics = self.model.evaluate(
            self._data_splits['test']['images'], self._data_splits['test']['labels'], verbose=self._verbosity
        )
        return self

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
        print(f'Data source: {", ".join(self._train_params.source_list)}')
        print(f'Number of interpolation types: {len(self._train_params.interpolation_list)}')
        print(f'\t{", ".join(self._train_params.interpolation_list)}')
        print(f'Number of chosen beings: {len(self._train_params.chosen_beings)}')
        print(f'\t{", ".join(self._train_params.chosen_beings)}')
        print(f'Number of classes: {len(self.class_names)}')
        print(f'\t{", ".join(self.class_names)}')
        print(f'Number of entries in dataset: {total_count}')
        print('=====================================================================')
        print(f'Number train images: {num_train_images} -> {num_train_labels}')
        sep = '\t'
        for event_idx, event_count in train_event_counts.iteritems():
            print(f'{sep}{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / num_train_labels:0.4f} %)', end='')
            sep = ', '
        print()
        print('=====================================================================')
        print(f'Number validation images: {num_val_images} -> {num_val_labels}')
        sep = '\t'
        for event_idx, event_count in val_event_counts.iteritems():
            print(f'{sep}{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / num_val_labels:0.4f} %)', end='')
            sep = ', '
        print()
        print('=====================================================================')
        print(f'Number test images: {num_test_images} -> {num_test_labels}')
        sep = '\t'
        for event_idx, event_count in test_event_counts.iteritems():
            print(f'{sep}{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / num_test_labels:0.4f} %)', end='')
            sep = ', '
        print()
        print('=====================================================================')
        print(f'Shape train images: {train_img_shape}')
        print(f'Shape validation images: {val_img_shape}')
        print(f'Shape test images: {test_img_shape}')
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
            plt.savefig(os.path.join(self.model_dir, f'samples_{title}_rows_{num_rows}_cols_{num_cols}.png'))
        else:
            plt.show()
        plt.close()
        return

    def display_metrics(self):
        if not self._eval_metrics:
            if not self._train_history:
                print('Model has not been trained')
            else:
                print('Model has not been evaluated')
            return
        print('==========================================================================')
        sep = '\n\t'
        for metric_index, metric_name in enumerate(self.model.metrics_names):
            print(f'{sep}{metric_name}: {self._eval_metrics[metric_index]:0.4f}', end='')
            sep = ', '
        print()
        print('==========================================================================')
        test_predictions = self.model.predict(
            self._data_splits['test']['images'], batch_size=self._train_params.batch_size
        )
        top_test_preds = [np.argmax(each_pred) for each_pred in test_predictions]
        annotate_conf = len(self.class_names) < 20
        self.__plot_confusion_matrix(
            y_pred=top_test_preds, y_true=self._data_splits['test']['labels'], class_labels=self.class_names,
            title=self._train_params.target_column, annotate_entries=annotate_conf, save_plot=True
        )

        for metric_name in self._train_history.params['metrics']:
            if not metric_name.startswith('val_'):
                self.__plot_history_metric(self._train_history, metric_name, save_plot=True)
        print('==========================================================================')
        return

    def __plot_confusion_matrix(self, y_true, y_pred, class_labels, title,
                                cmap: str = 'Blues', annotate_entries: bool = True, save_plot: bool = False):
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
            plt.savefig(os.path.join(self.model_dir, f'confusion_matrix_{title}.png'))
        else:
            plt.show()
        plt.close()
        return

    def __plot_history_metric(self, model_history, metric_name, save_plot: bool = False):
        style.use('ggplot')
        fig, ax = plt.subplots()

        ax.plot(model_history.epoch, model_history.history[metric_name],
                color='red', label=metric_name)
        ax.plot(model_history.epoch, model_history.history[f'val_{metric_name}'],
                color='black', label=f'validation {metric_name}')

        upper_y = np.max((model_history.history[metric_name], model_history.history[f'val_{metric_name}'])) * 1.2
        ax.set_ylim([0, upper_y])

        ax.set_xlim([model_history.epoch[0] - 0.5, model_history.epoch[-1] + 0.5])
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

    def display_model(self):
        print('==========================================================================')
        self.model.summary()
        """
        pip install pydot
        pip install pydotplus
        pip install graphviz
            https://www.graphviz.org/download/
            Make sure that the directory containing the dot executable is on your system's path
        """
        keras.utils.plot_model(
            self.model, os.path.join(self.model_dir, f'model_architecture.png'), show_shapes=True
        )
        print('==========================================================================')
        return

    def save_train_params(self):
        with open(os.path.join(self.model_dir, f'metadata.json'), 'w+') as meta_file:
            json.dump(self._train_params._asdict(), meta_file, indent=2)
        return

    def save_model(self):
        metric_dict = {}
        for metric_index, metric_name in enumerate(self.model.metrics_names):
            if isinstance(self._eval_metrics[metric_index], np.float32):
                metric_dict[metric_name] = float(self._eval_metrics[metric_index])
            else:
                metric_dict[metric_name] = self._eval_metrics[metric_index]

        with open(os.path.join(self.model_dir, f'metrics.json'), 'w+') as metric_file:
            json.dump(metric_dict, metric_file)
        self.model.save(os.path.join(self.model_dir, f'model_checkpoint.h5'))
        return


def main(margs):
    display_dataset = True
    display_samples = False
    display_metrics = True
    display_model = False
    #############################################

    heatmap_dir = margs.get('data_directory', os.path.join(DATA_DIR, 'heatmaps', 'Physio'))
    target_column = margs.get('target', 'event')
    num_subjects = margs.get('num_subjects', 20)
    num_epochs = margs.get('num_epochs', 20)
    batch_size = margs.get('batch_size', 16)
    learning_rate = margs.get('learning_rate', 1e-4)
    verbosity = margs.get('verbosity', 1)
    model_id = margs.get('model_id', '0')
    duration_list = ['epad_10', 'epad_20', 'epad_30', 'epad_40', 'epad_50']  # todo make argument

    #############################################
    rand_seed = 42
    padding_list = ['spad_10']
    interpolation_list = [
        Interpolation.LINEAR.name,
        Interpolation.QUADRATIC.name,
        Interpolation.CUBIC.name
    ]
    #############################################
    if not os.path.isdir(heatmap_dir):
        print(f'Dataset directory not found: {heatmap_dir}')
        return
    #############################################
    tf_classifier = TfClassifier(
        heatmap_dir,
        num_subjects=num_subjects, interpolation_types=interpolation_list,
        start_padding=padding_list, end_padding=duration_list, model_id=model_id,
        target=target_column, verbosity=verbosity, rand_seed=rand_seed,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate
    )

    tf_classifier.train()
    tf_classifier.evaluate()
    tf_classifier.save_model()

    if display_dataset:
        tf_classifier.display_dataset()
    if display_samples:
        tf_classifier.display_samples()
    if display_metrics:
        tf_classifier.display_metrics()
    if display_model:
        tf_classifier.display_model()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a Tensorflow classifier.')
    parser.add_argument('--data_directory', type=str, default=os.path.join(DATA_DIR, 'heatmaps', 'recorded'),
                        help='base directory or directory structure containing the image dataset')
    parser.add_argument('--target', type=str, default='event',
                        help='target variable to be used as the class label')
    parser.add_argument('--num_subjects', type=int, default=1,
                        help='number of subjects to use from the underlying datasource')
    parser.add_argument('--subject_name', type=str, default='Random',
                        help='Name of subject to use when training this classifier')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='maximum number of epochs over which to train the model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='size of a training batch')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate to use to train the model')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='verbosity level to use when reporting model updates: 0 -> off, 1-> on')
    parser.add_argument('--model_id', type=str, default='0',
                        help='string to use to label the model for future reference')

    args = parser.parse_args()
    main(vars(args))
