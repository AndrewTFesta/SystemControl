"""
@title
@description
"""
import os
import sys
from concurrent.futures.thread import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from bokeh.io import export_png
from bokeh.plotting import figure
from mne import events_from_annotations
from scipy.interpolate import interp1d
from selenium import webdriver
from tqdm import tqdm

from SystemControl import DATABASE_URL, DATA_DIR, CHROME_DRIVER_EXE
from SystemControl.DataSource import DataSource, SqlDb
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource, int_to_subject_str


class EventClassifier:
    CHROME_DRIVER = None

    def __init__(self, data_source: DataSource, subject: int = 1, spacing: int = 50,
                 start_padding: float = 0.1, end_padding: float = 0.1, duration: float = 0.5):
        self.data_source = data_source

        self._raw_data = None
        self._data = None
        self._events = None
        self._data_slices = None
        self._image_slices = None
        self.__init_data()

        self.subject = subject
        self.spacing = spacing

        self.start_padding = start_padding
        self.end_padding = end_padding
        self.duration = duration

        self.color_palette = 'Spectral11'
        return

    def __init_data(self):
        self._raw_data = self.data_source.get_mi_right_left(self.subject)
        self._data = self.data_source.get_data(self._raw_data)
        self._events = events_from_annotations(self._raw_data)

        # todo combine lists to not duplicate info
        self._data_slices = {}  # todo make into list of dicts
        self._image_slices = []
        return

    def set_data_source(self, data_source):
        self.data_source = data_source
        self.__init_data()
        return

    def set_subject(self, subject):
        self.subject = subject
        return

    def set_spacing(self, spacing):
        self.spacing = spacing
        return

    def set_start_padding(self, start_padding):
        self.start_padding = start_padding
        return

    def set_end_padding(self, end_padding):
        self.end_padding = end_padding
        return

    def set_duration(self, duration):
        self.duration = duration
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

    @staticmethod
    def __update_pbar_callback(future):
        future_arg = future.arg
        future_arg.update(1)
        return

    def save_images(self):
        # pbar = tqdm(total=len(self.image_slices), desc=f'{self.subject}: Saving images', file=sys.stdout)
        # with ThreadPoolExecutor(max_workers=50) as executor:
        #     for each_entry in self.image_slices:
        #         future = executor.submit(self.save_image_slice, each_entry)
        #         future.arg = pbar
        #         future.add_done_callback(self.__update_pbar_callback)
        # pbar.close()
        for each_entry in tqdm(self._image_slices, desc=f'{self.subject}: Saving images:', file=sys.stdout):
            self.save_image_slice(each_entry)
        return

    # noinspection PyTypeChecker
    def save_image_slice(self, slice_entry):
        slice_data = slice_entry['image']
        interp_type = slice_entry['interp']
        event_type = slice_entry['type']

        save_dir = os.path.join(
            DATA_DIR, 'heatmaps', f'{self.data_source.__str__()}',
            int_to_subject_str(self.subject), f'spacing_{self.spacing}', f'interp_{interp_type}', f'event_{event_type}'
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        spd_str = str(int(self.start_padding * 1000))
        epd_str = str(int(self.end_padding * 1000))
        dur_str = str(int(self.duration * 1000))
        out_fname = os.path.join(
            save_dir,
            f'img_spad_{spd_str}_epad_{epd_str}_dur_{dur_str}.png'
        )
        img_shape = slice_data.shape

        fig = figure(x_range=(0, img_shape[0]), y_range=(0, img_shape[1]))
        fig.image(
            image=[slice_data],
            x=0, y=0,
            dw=img_shape[0], dh=img_shape[1],
            palette=self.color_palette
        )
        fig.axis.visible = False
        fig.toolbar.logo = None
        fig.toolbar_location = None

        export_png(fig, filename=out_fname, webdriver=CHROME_DRIVER)
        return

    def build_all_images(self):
        for slice_type, slice_list in self._data_slices.items():
            for each_slice in tqdm(slice_list, desc=f'{self.subject}: Building images: {slice_type}', file=sys.stdout):
                self.slice2img(slice_type, each_slice)
        return

    def slice2img(self, slice_type: str, data_slice: np.ndarray):
        num_cols = (data_slice.shape[1] + 1) * self.spacing + data_slice.shape[1]
        lin_spaced_slice = []
        quad_spaced_slice = []
        cub_spaced_slice = []
        for each_row in data_slice:
            each_row = np.insert(each_row, 0, values=[0])
            each_row = np.append(each_row, 0)

            x = np.linspace(0, each_row.shape[0], num=each_row.shape[0], endpoint=False)
            xnew = np.linspace(0, each_row.shape[0] - 1, num=num_cols, endpoint=True)

            line_interp = interp1d(x, each_row, fill_value='extrapolate', kind='linear')
            quad_interp = interp1d(x, each_row, fill_value='extrapolate', kind='quadratic')
            cub_interp = interp1d(x, each_row, fill_value='extrapolate', kind='cubic')

            new_lin_y = line_interp(xnew)
            new_quad_y = quad_interp(xnew)
            new_cub_y = cub_interp(xnew)

            lin_spaced_slice.append(new_lin_y)
            quad_spaced_slice.append(new_quad_y)
            cub_spaced_slice.append(new_cub_y)

            # self.plot_slices(x, each_row, xnew, new_lin_y, new_quad_y, new_cub_y)
        self.prepare_slice(lin_spaced_slice, slice_type, 'linear')
        self.prepare_slice(quad_spaced_slice, slice_type, 'quad')
        self.prepare_slice(cub_spaced_slice, slice_type, 'cub')
        return

    @staticmethod
    def plot_slices(x_vals, row_vals, xnew, liny, quady, cuby):
        plt.plot(x_vals, row_vals, 'o')
        plt.plot(xnew, liny, '-')
        plt.plot(xnew, quady, '--')
        plt.plot(xnew, cuby, ':')

        plt.legend(['data', 'linear', 'quadratic', 'cubic'], loc='best')
        return

    def prepare_slice(self, np_slice, slice_type, interp):
        np_slice = np.array(np_slice)
        min_val = np.amin(np_slice)
        np_slice = np_slice + abs(min_val)
        max_val = np.amax(np_slice)
        np_slice = np_slice / max_val
        np_slice = np_slice.astype(np.float32)
        np_slice = np.transpose(np_slice)

        self._image_slices.append({
            'image': np_slice,
            'type': slice_type,
            'interp': interp
        })
        return

    def slice_data(self):
        freq = self._raw_data.info['sfreq']

        num_start_samples_padding = int(freq * self.start_padding)
        num_end_samples_padding = int(freq * self.end_padding)
        num_samples_per_event = int(freq * self.duration)

        event_list = self._events[0]
        event_types = self._events[1]
        d_transpose = np.transpose(self._data)
        for event_idx, each_event in enumerate(tqdm(event_list, desc=f'{self.subject}: Slicing data', file=sys.stdout)):
            sample_idx = each_event[0]
            event_id = each_event[2]
            event_str = EventClassifier.event_from_id(event_types, event_id)
            end_sample_idx = sample_idx + num_samples_per_event

            start_slice_idx = sample_idx - num_start_samples_padding
            end_slice_idx = end_sample_idx + num_end_samples_padding
            # only add to slice_list if data range is valid (non negative and not beyond bounds of d_transpose
            if start_slice_idx >= 0 and end_slice_idx < len(d_transpose):
                data_slice = d_transpose[start_slice_idx:end_slice_idx]
                if event_str not in self._data_slices:
                    self._data_slices[event_str] = []
                self._data_slices[event_str].append(data_slice)
            #     print(f'{start_slice_idx}:{end_slice_idx}, Event: {event_str}, Data points: {data_slice.shape}')
            # else:
            #     print(f'Data slice falls outside valid bounds ({0}:{len(d_transpose)}): '
            #           f'({start_slice_idx}:{end_slice_idx})')
        return

    def train(self):
        # todo
        return

    def predict(self):
        # todo
        return


def main():
    # todo parallelize - slice_data()
    # todo parallelize - build_all_images()
    # todo parallelize - save_images()
    row_spacing: int = 100
    start_padding: float = 0.1
    end_padding: float = 0.1
    duration: float = 0.5

    db_path = DATABASE_URL
    database = SqlDb.SqlDb(db_path)
    physio_data_source = PhysioDataSource(database)

    for each_subject in PhysioDataSource.SUBJECT_NUMS:
        try:
            event_classifier = EventClassifier(
                physio_data_source, subject=each_subject, spacing=row_spacing,
                start_padding=start_padding, end_padding=end_padding, duration=duration
            )
            event_classifier.slice_data()
            event_classifier.build_all_images()
            event_classifier.save_images()
        except Exception as e:
            print(str(e))
    return


if __name__ == '__main__':
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    CHROME_DRIVER = webdriver.Chrome(CHROME_DRIVER_EXE, options=options)
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
