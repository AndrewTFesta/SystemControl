"""
@title
@description
"""
import os
import shutil
import sys
import time
from enum import Enum, auto
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyvips
import seaborn as sns
from scipy.interpolate import interp1d
from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource
from SystemControl.utilities import find_files_by_type

dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


def plot_sample(xold, yold, xnew, ynew, type_str):
    plt.plot(xold, yold, 'o')
    plt.plot(xnew, ynew, '-')

    plt.legend(['data', type_str], loc='best')
    plt.show()
    plt.close()
    return


class CMAP(Enum):
    rocket_r = 0
    YlGnBu = 1


class Interpolation(Enum):
    LINEAR = 0
    QUADRATIC = 1
    CUBIC = 2


class DataTransformer:

    def __init__(self, data_source: DataSource, subject: str = '', spacing: int = 50, cmap: CMAP = CMAP.rocket_r,
                 interpolation: Interpolation = Interpolation.LINEAR, start_padding: float = 0.1,
                 duration: float = 0.5, timing_resolution: float = 100., debug: bool = False):
        self.data_source = data_source

        self._timing_resolution = timing_resolution
        self._subject = subject
        self._spacing = spacing
        self._cmap = cmap
        self._interpolation = interpolation
        self._start_padding = int(start_padding * self._timing_resolution) / self._timing_resolution
        self._duration = int(duration * self._timing_resolution) / self._timing_resolution

        self._data = None
        self._events = None
        self._tqdm_base_desc = None

        self.base_dir = None
        self.data_slices = None
        self.image_slices = None

        self.__init_data()

        self._debug = debug
        self._debug_len = 5
        if debug:
            self._spacing = 5
        return

    def __init_data(self):
        if self.data_source and self._subject in self.data_source.subject_names:
            self._data = self.data_source.get_data()  # todo set subject in datasource
            self._events = self.data_source.get_events()

            self.base_dir = os.path.join(
                DATA_DIR, 'heatmaps',
                self.data_source.name,
                f'spad_{int(self._start_padding * self._timing_resolution)}',
                f'duration_{int(self._duration * self._timing_resolution)}',
                self._interpolation.name,
                self._subject
            )
            self.data_slices = []
        return

    def __entry_id(self, data_entry: np.ndarray, sample_idx: int, event_str: str):
        entry_str = f'{str(data_entry)}_{str(sample_idx)}_{event_str}_' \
                    f'{self._subject}_{self._spacing}_{self._cmap}_{self._interpolation.name}' \
                    f'{self._start_padding}_{self._duration}'
        entry_hash = hash(entry_str)
        return abs(entry_hash)

    def set_data_source(self, data_source):
        self.data_source = data_source
        self.__init_data()
        return

    def set_subject(self, subject):
        self._subject = subject
        self.__init_data()
        return

    def set_spacing(self, spacing):
        self._spacing = spacing
        self.__init_data()
        return

    def set_cmap(self, cmap):
        self._cmap = cmap
        self.__init_data()
        return

    def set_interpolation(self, interpolation):
        self._interpolation = interpolation
        self.__init_data()
        return

    def set_start_padding(self, start_padding):
        self._start_padding = start_padding
        self.__init_data()
        return

    def set_duration(self, duration):
        self._duration = duration
        self.__init_data()
        return

    def save_images(self, use_pyvips=True):
        if os.path.isdir(self.base_dir):
            shutil.rmtree(self.base_dir)
        for each_entry in self.data_slices:
            self.save_image_slice(each_entry, use_pyvips)
        return

    def save_image_slice(self, slice_entry, use_pyvips=True):
        # todo parameterize save location
        # todo force overwrite
        image_slice = slice_entry['image_slice']
        event_type = slice_entry['type']
        image_id = slice_entry['id']
        save_dir = os.path.join(self.base_dir, f'event_{event_type}')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        out_fname = os.path.join(save_dir, f'{image_id}.png')
        heatmap = self.build_heatmap(image_slice)
        # todo check speed of tensorflow img functions
        # opencv faster than pillow: 2.63 it/s PIL vs 3.48 it/s CV2
        if use_pyvips:
            height, width, bands = heatmap.shape
            linear_heatmap = heatmap.reshape(width * height * bands)
            vi = pyvips.Image.new_from_memory(
                # linear_heatmap.data, width, height, bands, dtype_to_format[str(heatmap.dtype)]
                linear_heatmap.data, width, height, bands, 'uchar'
            )
            vi.write_to_file(out_fname)
        else:
            cv2.imwrite(out_fname, heatmap)
        return heatmap

    def build_heatmap(self, image_slice):
        heat_map = sns.heatmap(image_slice, xticklabels=False, yticklabels=False, cmap=self._cmap.name, cbar=False)
        sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)

        fig = heat_map.get_figure()
        fig.tight_layout(pad=0)
        fig.canvas.draw()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image_from_plot

    def build_all_images(self):
        for each_entry in self.data_slices:
            self.slice2img(each_entry)
        return

    @staticmethod
    def __interpolate_row(row_sample, num_rows, interp_type):
        # add a zero before interpolation to lock either end to ground state
        each_sample = np.insert(row_sample, 0, values=[0])
        each_sample = np.append(each_sample, 0)

        x = np.linspace(0, each_sample.shape[0], num=each_sample.shape[0], endpoint=False)
        xnew = np.linspace(0, each_sample.shape[0] - 1, num=num_rows, endpoint=True)

        interp_func = interp1d(x, each_sample, fill_value='extrapolate', kind=interp_type)
        new_y = interp_func(xnew)
        return new_y

    def slice2img(self, data_entry: dict):
        # todo make except 2d array of values
        data_slice = data_entry['slice']
        data_list = [each_sample['data'] for each_sample in data_slice]
        np_data = np.array(data_list)

        # place one row between each and one at either end
        # add number of rows once more to account for each signal
        num_rows = (np_data.shape[1] + 1) * self._spacing + np_data.shape[1]
        interp_str = self._interpolation.name.lower()
        partial_interpolate = partial(self.__interpolate_row, num_rows=num_rows, interp_type=interp_str)

        spaced_slice = list(map(partial_interpolate, np_data))
        data_entry['image_slice'] = np.transpose(spaced_slice)
        return spaced_slice

    def slice_data(self):
        freq = self.data_source.sample_freq
        num_start_samples_padding = int(freq * self._start_padding)
        num_samples_per_event = int(freq * self._duration)

        event_list = self._events
        if self._debug:
            event_list = event_list[:self._debug_len]

        for each_event in event_list:
            new_slice = self.extract_slice(each_event, self._data, num_samples_per_event, num_start_samples_padding)
            if new_slice:
                self.data_slices.append(new_slice)
        return

    def extract_slice(self, event_info, data, num_samples_per_event, num_start_samples_padding):
        sample_idx = event_info['idx']
        event_str = event_info['event']

        start_sample_idx = sample_idx - num_start_samples_padding
        end_sample_idx = sample_idx + num_samples_per_event

        # only add to slice_list if data range is valid (non negative and not beyond bounds of d_transpose
        slice_entry = {}
        if start_sample_idx >= 0 and end_sample_idx < len(data):
            data_slice = data[start_sample_idx:end_sample_idx]
            slice_entry = {
                'id': self.__entry_id(data_slice, sample_idx, event_str),
                'slice': data_slice,
                'image_slice': None,
                'type': event_str
            }
        return slice_entry


def main():
    debug = False

    row_spacing: int = 100
    interp: Interpolation = Interpolation.LINEAR

    start_pad_points = 1
    start_pad_step = 0.1

    duration_points = 5
    duration_step = 0.1

    data_source_list = [PhysioDataSource()]
    start_padding_list: list = [(idx + 1) * start_pad_step for idx in range(0, start_pad_points)]
    duration_list = [(idx + 1) * duration_step for idx in range(0, duration_points)]
    enum_members = list(Interpolation.__members__.values())
    valid_subject_names = data_source_list[0].subject_names

    # todo data only relies on data_source and subject
    # todo slicing only relies on spad and duration
    # todo image creation only relies on interpolation

    num_calls = len(data_source_list)
    num_calls *= len(start_padding_list)
    num_calls *= len(duration_list)
    num_calls *= len(enum_members)
    num_calls *= len(valid_subject_names)

    timings_list = []
    pbar = tqdm(total=num_calls, desc=f'', file=sys.stdout)
    for each_data_source in data_source_list:
        data_transformer = DataTransformer(
            each_data_source, subject=valid_subject_names[0], spacing=row_spacing, cmap=CMAP.rocket_r,
            interpolation=interp,
            start_padding=start_padding_list[0], duration=duration_list[0], debug=debug
        )

        for each_start in start_padding_list:
            data_transformer.set_start_padding(each_start)
            for each_duration in duration_list:
                data_transformer.set_duration(each_duration)
                for each_enum in enum_members:
                    data_transformer.set_interpolation(each_enum)
                    for each_subject in valid_subject_names:
                        try:
                            data_transformer.set_subject(each_subject)
                            base_desc = f'{each_data_source.name}: {each_start}: {each_duration}: ' \
                                        f'{each_enum}: {each_subject}'

                            start_time = time.time()
                            pbar.set_description(f'{base_desc}: Validating existing images')
                            img_paths = find_files_by_type('png', root_dir=data_transformer.base_dir)
                            num_exist_imgs = len(img_paths)

                            pbar.set_description(f'{base_desc}: Slicing data')
                            data_transformer.slice_data()
                            if num_exist_imgs > len(data_transformer.data_slices):
                                shutil.rmtree(data_transformer.base_dir)
                                num_exist_imgs = 0

                            if num_exist_imgs < len(data_transformer.data_slices):
                                pbar.set_description(f'{base_desc}: Building images')
                                data_transformer.build_all_images()
                                pbar.set_description(f'{base_desc}: Saving images')
                                data_transformer.save_images()

                            end_time = time.time()
                            d_time = end_time - start_time
                            timings_list.append(d_time)
                        except Exception as e:
                            print(str(e))
                        finally:
                            pbar.update(1)
    pbar.close()

    out_dir = os.path.join(DATA_DIR, 'output', 'timings')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    timing_fname = os.path.join(out_dir, f'dataset_generation_{num_calls}.txt')
    with open(timing_fname, 'w+') as timing_file:
        for each_timing in timings_list:
            timing_file.write(f'{each_timing}\n')
    return


if __name__ == '__main__':
    main()

########################################
#
# # Define a monte-carlo cross-validation generator (reduce variance):
# scores = []
# epochs_data = epochs.get_data()
# epochs_data_train = epochs_train.get_data()
# cv = ShuffleSplit(10, test_size=0.2, random_state=42)
# cv_split = cv.split(epochs_data_train)
#
# # Assemble a classifier
# lda = LinearDiscriminantAnalysis()
# csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
#
# # Use scikit-learn Pipeline with cross_val_score function
# clf = Pipeline([('CSP', csp), ('LDA', lda)])
# scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
#
# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance)
# print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))
#
# # plot CSP patterns estimated on full data for visualization
# csp.fit_transform(epochs_data, labels)
#
# layout = read_layout('EEG1005')
# csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg', units='Patterns (AU)', size=1.5)
#
# ########################################
#
# sfreq = raw.info['sfreq']
# w_length = int(sfreq * 0.5)  # running classifier: window length
# w_step = int(sfreq * 0.1)  # running classifier: window step size
# w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
#
# scores_windows = []
#
# for train_idx, test_idx in cv_split:
#     y_train, y_test = labels[train_idx], labels[test_idx]
#
#     X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
#     X_test = csp.transform(epochs_data_train[test_idx])
#
#     # fit classifier
#     lda.fit(X_train, y_train)
#
#     # running classifier: test classifier on sliding window
#     score_this_window = []
#     for sigs in w_start:
#         X_test = csp.transform(epochs_data[test_idx][:, :, sigs:(sigs + w_length)])
#         score_this_window.append(lda.score(X_test, y_test))
#     scores_windows.append(score_this_window)
#
# # Plot scores over time
# w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
#
# plt.figure()
# plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
# plt.axvline(0, linestyle='--', color='k', label='Onset')
# plt.axhline(0.5, linestyle='-', color='k', label='Chance')
# plt.xlabel('time (s)')
# plt.ylabel('classification accuracy')
# plt.title('Classification score over time')
# plt.legend(loc='lower right')
# plt.show()
# return edf_data_list
#
