"""
@title
@description
"""
import json
import os
import random
import time
import collections

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource, int_to_subject_str
from SystemControl.DataTransformer import Interpolation
from SystemControl.utilities import find_files_by_type, filter_list_of_dicts, find_files_by_name


def show_sample(img_sample, target_label):
    plt.figure()
    plt.imshow(img_sample)
    plt.tick_params(
        axis='both', which='both',
        bottom=False, top=False, labelbottom=False,
        right=False, left=False, labelleft=False
    )
    plt.xlabel(target_label)
    plt.grid(False)
    plt.show()
    plt.close()
    return


def show_top_samples(sample_imgs, sample_labels, title: str, num_rows: int = 5, num_cols: int = 5):
    fig, ax_list = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    fig.suptitle(f'{title} images: top {num_rows * num_cols}')
    for each_ax_idx, each_ax in enumerate(ax_list.flat):
        each_ax.set_xticks([])
        each_ax.set_yticks([])
        each_ax.imshow(sample_imgs[each_ax_idx])
        each_ax.set_xlabel(sample_labels[each_ax_idx])
    plt.show()
    plt.close()
    return


def plot_confusion_matrix(y_true, y_pred, class_labels, title, cmap='Blues', annotate_entries=True):
    title = f'Target: \'{title}\''
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

    ax.set_title(title)
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

    plt.show()
    plt.close()
    return


def plot_history_metric(model_history, metric_name):
    upper_y = np.max((model_history.history[metric_name], model_history.history[f'val_{metric_name}'])) * 1.2
    plt.plot(model_history.epoch, model_history.history[metric_name], label=metric_name)
    plt.plot(model_history.epoch, model_history.history[f'val_{metric_name}'], label=f'validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.ylim([0, upper_y])
    plt.xlim([model_history.epoch[0] - 0.5, model_history.epoch[-1] + 0.5])
    plt.legend(loc='lower right')
    plt.title = f'Model history: \'{metric_name}\''

    # todo save model
    plt.show()
    plt.close()
    return


TrainParameters = collections.namedtuple(
    'TrainParameter',
    [
        'chosen_beings', 'interpolation_list', 'source_list', 'target_column',
        'start_padding', 'end_padding', 'img_dims', 'learning_rate', 'batch_size', 'num_epochs'
    ]
)


class TfClassifier:

    def __init__(self, base_data_directory, chosen_beings: list, interpolation_types: list, sources: list,
                 start_padding: list, end_padding: list, target: str = 'event', mdl_id: str = None,
                 verbosity: int = 1, learning_rate: float = 1e-4, num_epochs: int = 20, batch_size: int = 1,
                 resize_height: int = 256, resize_width: int = 224, rand_seed: int = 42):
        time_stamp = time.strftime('%y_%m_%d_%H_%M_%S', time.gmtime())
        self.model_name = f'tf_model'
        if mdl_id:
            self.model_name = f'_{mdl_id}'

        self.model_dir = os.path.join(DATA_DIR, 'models', 'tf', f'{self.model_name}_{time_stamp}')

        self._train_params = TrainParameters(
            chosen_beings=chosen_beings, interpolation_list=interpolation_types, source_list=sources,
            start_padding=start_padding, end_padding=end_padding,
            target_column=target, img_dims=[resize_width, resize_height], learning_rate=learning_rate,
            batch_size=batch_size, num_epochs=num_epochs
        )
        self._rand_seed = rand_seed
        self._verbosity = verbosity

        self._data_directory = base_data_directory
        self._raw_data = self.__load_image_files()
        self._filtered_dataframe, self.class_names = self.__filter_data_entries()
        self._data_splits = self.__split_data()

        self.model = self.__build_cnn_v0()

        self._train_history = None
        self._eval_metrics = None
        return

    def __load_image_files(self):
        img_files = find_files_by_type('png', self._data_directory)
        data_list = []
        for each_file in img_files:
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

    def __filter_data_entries(self):
        filter_criteria = {
            'subject': self._train_params.chosen_beings,
            'interpolation': self._train_params.interpolation_list,
            'source': self._train_params.source_list,
            'end_padding': self._train_params.end_padding
        }
        filtered_dataset = filter_list_of_dicts(self._raw_data, filter_criteria)
        rand_generator = random.Random(self._rand_seed)
        rand_generator.shuffle(filtered_dataset)
        filtered_df = pd.DataFrame(filtered_dataset)
        class_names = sorted(filtered_df[self._train_params.target_column].unique())
        target_str_to_idx = {class_name: class_idx for class_idx, class_name in enumerate(class_names)}
        filtered_df.replace(target_str_to_idx, inplace=True)
        return filtered_df, class_names

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
        model.add(keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=self.default_model_optimizer(self._train_params.learning_rate),
            loss=self.default_model_loss(),
            metrics=self.default_model_metrics()
        )
        return model

    @staticmethod
    def get_model_callbacks():
        fit_callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', verbose=1, patience=2, mode='auto', restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='{epoch:02d}-{val_loss:.2f}.hdf5',
                monitor='val_loss', verbose=1, mode='auto', save_best_only=True
            )
        ]
        return fit_callbacks

    @staticmethod
    def default_model_metrics():
        metrics_list = [
            keras.metrics.SparseCategoricalAccuracy(name='SparseCategoricalAccuracy'),
            # keras.metrics.CosineSimilarity(name='cos_similarity'),
            # keras.metrics.CategoricalHinge(name='categorical_hinge'),
            # keras.metrics.MeanAbsoluteError(name='mae'),
            # keras.metrics.MeanSquaredError(name='mse'),
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

    def __split_data(self, test_size=0.25, val_size=0.15):
        train_val_df, test_filtered_df = train_test_split(
            self._filtered_dataframe, test_size=test_size, random_state=self._rand_seed
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
            callbacks=self.get_model_callbacks(),
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

        train_images = self._data_splits['train']['images']
        val_images = self._data_splits['validation']['images']
        test_images = self._data_splits['test']['images']

        train_labels = self._data_splits['train']['labels']
        val_labels = self._data_splits['validation']['labels']
        test_labels = self._data_splits['test']['labels']

        print('=====================================================================')
        print(f'Data source: {", ".join(self._train_params.source_list)}')
        print(f'Number of interpolation types: {len(self._train_params.interpolation_list)}')
        print(f'\t{", ".join(self._train_params.interpolation_list)}')
        print(f'Number of chosen beings: {len(self._train_params.chosen_beings)}')
        print(f'\t{", ".join(self._train_params.chosen_beings)}')
        print(f'Number of classes: {len(self.class_names)}')
        print(f'\t{", ".join(self.class_names)}')
        print(f'Number of entries in filtered dataset: {len(self._filtered_dataframe)}')
        print('=====================================================================')
        print(f'Number train images: {len(train_images)} -> {len(train_labels)}')
        sep = '\t'
        for event_idx, event_count in train_event_counts.iteritems():
            print(f'{sep}{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / len(train_labels):0.4f} %)', end='')
            sep = ', '
        print()
        print('=====================================================================')
        print(f'Number validation images: {len(val_images)} -> {len(val_labels)}')
        sep = '\t'
        for event_idx, event_count in val_event_counts.iteritems():
            print(f'{sep}{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / len(val_labels):0.4f} %)', end='')
            sep = ', '
        print()
        print('=====================================================================')
        print(f'Number test images: {len(test_images)} -> {len(test_labels)}')
        sep = '\t'
        for event_idx, event_count in test_event_counts.iteritems():
            print(f'{sep}{self.class_names[event_idx]}: {event_count} '
                  f'({(event_count * 100) / len(test_labels):0.4f} %)', end='')
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

        show_top_samples(train_images, train_labels, title='Train', num_rows=5, num_cols=5)
        show_top_samples(val_images, val_labels, title='Validation', num_rows=5, num_cols=5)
        show_top_samples(test_images, test_labels, title='Test', num_rows=5, num_cols=5)
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
        plot_confusion_matrix(
            y_pred=top_test_preds, y_true=self._data_splits['test']['labels'], class_labels=self.class_names,
            title=self._train_params.target_column, annotate_entries=False
        )

        # todo iterate through all metrics
        plot_history_metric(self._train_history, 'SparseCategoricalAccuracy')
        plot_history_metric(self._train_history, 'loss')
        print('==========================================================================')
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
            self.model, os.path.join(self._output_dir, f'model_architecture.png'), show_shapes=True
        )
        print('==========================================================================')
        return

    def save_model(self):
        meta_dict = {
            'train_params': self._train_params._asdict(),
            'name': self.model_name,
        }
        metric_dict = {}
        for metric_index, metric_name in enumerate(self.model.metrics_names):
            if isinstance(self._eval_metrics[metric_index], np.float32):
                metric_dict[metric_name] = float(self._eval_metrics[metric_index])
            else:
                metric_dict[metric_name] = self._eval_metrics[metric_index]

        if not os.path.isdir(self._output_dir):
            os.makedirs(self._output_dir)

        with open(os.path.join(self._output_dir, f'metadata.json'), 'w+') as meta_file:
            json.dump(meta_dict, meta_file)

        with open(os.path.join(self._output_dir, f'metrics.json'), 'w+') as metric_file:
            json.dump(metric_dict, metric_file)
        self.model.save(os.path.join(self._output_dir, f'model_checkpoint.h5'))
        return

    def load_pretrained(self, base_dir=None):
        if not base_dir:
            base_dir = os.path.join(DATA_DIR, 'output', 'tf')

        if not os.path.isdir(base_dir):
            print(f'Directory does not exist: {base_dir}')
            return False

        poss_model_dir = sorted([
            os.path.join(base_dir, each_dir)
            for each_dir in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, each_dir))
        ])
        if len(poss_model_dir) == 0:
            print(f'Unable to locate model files in directory: {base_dir}')
            return False

        model_dir = poss_model_dir[-1]
        model_state_fname = find_files_by_type('h5', root_dir=model_dir)[-1]
        meta_fname = find_files_by_name('metadata', root_dir=model_dir)[-1]
        metrics_fname = find_files_by_name('metrics', root_dir=model_dir)[-1]

        if meta_fname and model_state_fname:
            self.model = keras.models.load_model(model_state_fname)

            with open(meta_fname, 'r') as meta_file:
                model_info = json.load(meta_file)
                self.model_name = model_info['name']
                self._output_dir = os.path.join(DATA_DIR, 'output', 'tf', self.model_name)
                self._train_params = TrainParameters(**model_info['train_params'])

            with open(metrics_fname, 'r') as metric_file:
                model_info = json.load(metric_file)
                self._eval_metrics = (model_info.values())
                print(self._eval_metrics)
        else:
            print(f'Unable to locate model files in directory: {base_dir}')
            return False
        return True


def main():
    verbosity = 1
    load_model = False
    train_model = True

    display_dataset = True
    display_samples = False
    display_metrics = True
    display_model = False
    #############################################

    heatmap_dir = os.path.join(DATA_DIR, 'heatmaps')
    target_column = 'event'
    num_subjects = 20
    num_epochs = 10
    b_size = 1
    lr = 1e-4

    #############################################
    physio_ds = PhysioDataSource()
    rand_seed = 42
    rand_generator = random.Random(rand_seed)
    potential_candidates = physio_ds.subject_names
    chosen_beings = sorted(rand_generator.sample(potential_candidates, k=num_subjects))

    padding_list = ['spad_10']
    duration_list = ['epad_10']
    source_list = [physio_ds.name]
    interpolation_list = [
        Interpolation.LINEAR.name,
        Interpolation.QUADRATIC.name,
        Interpolation.CUBIC.name
    ]
    #############################################
    tf_classifier = TfClassifier(
        heatmap_dir,
        chosen_beings=chosen_beings, interpolation_types=interpolation_list, sources=source_list,
        start_padding=padding_list, end_padding=duration_list,
        target=target_column, preload=load_model, verbosity=verbosity,
        num_epochs=num_epochs, batch_size=b_size, learning_rate=lr
    )
    if train_model:
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
    main()
