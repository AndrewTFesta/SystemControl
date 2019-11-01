"""
@title
@description
"""
import csv
import os
import sys
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mne import events_from_annotations
from scipy.interpolate import interp1d
from tqdm import tqdm

from SystemControl import DATABASE_URL, DATA_DIR
from SystemControl.DataSource import DataSource, SqlDb
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource, int_to_subject_str, SUBJECT_NUMS


class CMAP(Enum):
    rocket_r = 0
    YlGnBu = 1


class Interpolation(Enum):
    LINEAR = 0
    QUADRATIC = 1
    CUBIC = 2


class DataTransformer:

    def __init__(self, data_source: DataSource, subject: int = 1, spacing: int = 50, cmap: CMAP = CMAP.rocket_r,
                 interpolation: Interpolation = Interpolation.LINEAR, start_padding: float = 0.1, duration: float = 0.5,
                 debug: bool = False):
        self.data_source = data_source

        self._subject = subject
        self._spacing = spacing
        self._cmap = cmap
        self._interpolation = interpolation
        self._start_padding = int(start_padding * 100) / 100.
        self._duration = int(duration * 100) / 100.

        self._raw_data = None
        self._data = None
        self._events = None
        self._data_slices = None
        self._image_slices = None
        self._base_dir = None
        self._tqdm_base_desc = None

        self.__init_data()

        self._debug = debug
        self._debug_len = 5
        return

    def __init_data(self):
        if self.data_source and PhysioDataSource.validate_subject_num(self._subject):
            self._raw_data = self.data_source.get_mi_right_left(self._subject)
            self._data = self.data_source.get_data(self._raw_data)
            self._events = events_from_annotations(self._raw_data)
            self._base_dir = os.path.join(
                DATA_DIR, 'heatmaps',
                self.data_source.__str__(),
                str(int(self._start_padding * 100)),
                str(int(self._duration * 100)),
                self._interpolation.name,
                int_to_subject_str(self._subject)
            )
            self._tqdm_base_desc = f'{self._subject}: {self._interpolation.name}'
            self._data_slices = []
        return

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

    @staticmethod
    def event_from_id(event_dict, event_id):
        for evt_key, evt_val in event_dict.items():
            if evt_val == event_id:
                evt_str = evt_key
                break
        else:
            evt_str = None
        return evt_str

    def __entry_id(self, data_entry: np.ndarray, sample_idx: int, event_str: str):
        entry_str = f'{str(data_entry)}_{str(sample_idx)}_{event_str}_' \
                    f'{self._subject}_{self._spacing}_{self._cmap}_{self._interpolation.name}' \
                    f'{self._start_padding}_{self._duration}'
        entry_hash = hash(entry_str)
        return abs(entry_hash)

    def save_metadata(self):
        if not os.path.isdir(self._base_dir):
            os.makedirs(self._base_dir)
        out_fname = os.path.join(self._base_dir, f'metadata_{int_to_subject_str(self._subject)}.csv')
        entry_list = []
        for each_entry in self._data_slices:
            entry_dict = {
                'id': each_entry['id'],
                'type': each_entry['type'],
                'spacing': self._spacing,
                'color_map': self._cmap.name,
                'start_padding': self._start_padding,
                'duration': self._duration,
                'interpolation': self._interpolation.name,
            }
            entry_list.append(entry_dict)
        with open(out_fname, 'w+', newline='') as metafile:
            w = csv.DictWriter(metafile, entry_list[-1].keys())
            w.writeheader()
            w.writerows(entry_list)
        return

    def save_images(self):
        pbar = tqdm(total=len(self._data_slices), desc=f'{self._tqdm_base_desc}: Saving images', file=sys.stdout)
        for each_entry in self._data_slices:
            self.save_image_slice(each_entry)
            pbar.update(1)
        pbar.close()
        return

    # noinspection PyTypeChecker
    def save_image_slice(self, slice_entry):
        image_slice = slice_entry['image_slice']
        event_type = slice_entry['type']
        image_id = slice_entry['id']
        save_dir = os.path.join(self._base_dir, f'event_{event_type}')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        out_fname = os.path.join(save_dir, f'{image_id}.png')
        heatmap = self.build_heatmap(image_slice)
        cv2.imwrite(out_fname, heatmap)  # opencv faster than pillow: 2.63 it/s PIL vs 3.48 it/s CV2
        return

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
        pbar = tqdm(total=len(self._data_slices), desc=f'{self._tqdm_base_desc}: Building images', file=sys.stdout)
        for each_entry in self._data_slices:
            self.slice2img(each_entry)
            pbar.update(1)
        pbar.close()
        return

    def slice2img(self, data_entry: dict, plot_interpolation: bool = False):
        data_slice = data_entry['slice']
        interp_str = self._interpolation.name.lower()
        num_cols = (data_slice.shape[1] + 1) * self._spacing + data_slice.shape[1]
        lin_spaced_slice = []
        for each_row in data_slice:
            each_row = np.insert(each_row, 0, values=[0])
            each_row = np.append(each_row, 0)

            x = np.linspace(0, each_row.shape[0], num=each_row.shape[0], endpoint=False)
            xnew = np.linspace(0, each_row.shape[0] - 1, num=num_cols, endpoint=True)

            line_interp = interp1d(x, each_row, fill_value='extrapolate', kind=interp_str)
            new_lin_y = line_interp(xnew)
            lin_spaced_slice.append(new_lin_y)

            if plot_interpolation:
                self.plot_interpolation(x, each_row, xnew, new_lin_y, self._interpolation.name)
        data_entry['image_slice'] = lin_spaced_slice
        return lin_spaced_slice

    @staticmethod
    def plot_interpolation(x_vals, row_vals, xnew, liny, type_str):
        plt.plot(x_vals, row_vals, 'o')
        plt.plot(xnew, liny, '-')

        plt.legend(['data', type_str], loc='best')
        return

    @staticmethod
    def normalize_slice(np_slice):
        np_slice = np.array(np_slice)
        min_val = np.amin(np_slice)
        np_slice = np_slice + abs(min_val)
        max_val = np.amax(np_slice)
        np_slice = np_slice / max_val
        np_slice = np_slice.astype(np.float32)
        np_slice = np.transpose(np_slice)
        return np_slice

    def slice_data(self):
        freq = self._raw_data.info['sfreq']

        num_start_samples_padding = int(freq * self._start_padding)
        num_samples_per_event = int(freq * self._duration)

        event_list = self._events[0]
        if self._debug:
            event_list = event_list[:self._debug_len]
        event_types = self._events[1]
        d_transpose = np.transpose(self._data)

        pbar = tqdm(total=len(event_list), desc=f'{self._tqdm_base_desc}: Slicing data', file=sys.stdout)
        for event_idx, each_event in enumerate(event_list):
            self.extract_slice(
                each_event, event_types, num_samples_per_event, num_start_samples_padding, d_transpose
            )
            pbar.update(1)
        pbar.close()
        return

    def extract_slice(self, slice_event, event_types, num_samples_per_event, num_start_samples_padding, data):
        sample_idx = slice_event[0]
        event_id = slice_event[2]
        event_str = DataTransformer.event_from_id(event_types, event_id)
        start_sample_idx = sample_idx - num_start_samples_padding
        end_sample_idx = sample_idx + num_samples_per_event

        # only add to slice_list if data range is valid (non negative and not beyond bounds of d_transpose
        if start_sample_idx >= 0 and end_sample_idx < len(data):
            data_slice = data[start_sample_idx:end_sample_idx]
            slice_entry = {
                'id': self.__entry_id(data_slice, sample_idx, event_str),
                'slice': data_slice,
                'image_slice': None,
                'type': event_str
            }
            self._data_slices.append(slice_entry)
        return


def main():
    # todo track time to run functions
    # todo only load data that is missing
    # todo validate dataset
    row_spacing: int = 100
    interp: Interpolation = Interpolation.LINEAR
    start_padding_list: list = [0.1]
    duration_list: list = [0.5]
    debug = False

    db_path = DATABASE_URL
    database = SqlDb.SqlDb(db_path)
    physio_data_source = PhysioDataSource(database)

    data_transformer = DataTransformer(
        physio_data_source, subject=1, spacing=row_spacing, cmap=CMAP.rocket_r, interpolation=interp,
        start_padding=start_padding_list[0], duration=duration_list[0], debug=debug
    )

    for each_subject in SUBJECT_NUMS:
        for each_enum in Interpolation:
            for each_start in start_padding_list:
                for each_duration in duration_list:
                    try:
                        data_transformer.set_subject(each_subject)
                        data_transformer.set_interpolation(each_enum)
                        data_transformer.set_start_padding(each_start)
                        data_transformer.set_start_padding(each_duration)

                        data_transformer.slice_data()
                        data_transformer.build_all_images()
                        data_transformer.save_images()
                        data_transformer.save_metadata()
                    except Exception as e:
                        print(str(e))
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
