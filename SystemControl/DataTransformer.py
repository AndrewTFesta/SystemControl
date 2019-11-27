"""
@title
@description
"""
import hashlib
import os
import sys
import time
from enum import Enum, auto

import cv2
import matplotlib
import matplotlib.cm
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource

matplotlib.use('Qt5Agg')
style.use('ggplot')


class CMAP(Enum):
    rocket_r = auto()
    YlGnBu = auto()
    Spectral = auto()
    Blues = auto()
    BuGn = auto()
    BuPu = auto()
    Greens = auto()
    OrRd = auto()
    Oranges = auto()
    PuBu = auto()
    Reds = auto()
    gist_heat = auto()


class Interpolation(Enum):
    LINEAR = 0
    QUADRATIC = 1
    CUBIC = 2


def interpolate_row(row_sample: np.ndarray, num_rows: int, interp_type: Interpolation):
    interp_str = interp_type.name.lower()
    # add a zero before interpolation to lock either end to ground state
    each_sample = np.insert(row_sample, 0, values=[0])
    each_sample = np.append(each_sample, 0)

    x = np.linspace(0, each_sample.shape[0], num=each_sample.shape[0], endpoint=False)
    xnew = np.linspace(0, each_sample.shape[0] - 1, num=num_rows, endpoint=True)

    interp_func = interp1d(x, each_sample, fill_value='extrapolate', kind=interp_str)
    new_y = interp_func(xnew)
    return new_y


def plot_interpolation(xnew, ynew, interp_type):
    fig = plt.figure(figsize=(12, 12))
    axes = fig.add_subplot(1, 1, 1)
    axes.plot(xnew, ynew, color='navy')

    axes.set_title(f'{interp_type.name} interpolation of signal')
    axes.set_xlabel('X-range')
    axes.set_ylabel('Interpolated values')

    plt.show()
    plt.close()
    return


def img_id(data_entry: np.ndarray):
    entry_hash = hashlib.sha3_256(data_entry.tobytes()).hexdigest()
    return entry_hash


class DataTransformer:

    def __init__(self, data_source: DataSource, window_length: float, num_rows: int = 100):
        self.data_source = data_source
        self.window_length = window_length

        self.cmap = matplotlib.cm.get_cmap(CMAP.gist_heat.name)
        self.num_rows = num_rows
        self.num_cols = int(data_source.sample_freq * window_length)

        self.base_dir = os.path.join(
            DATA_DIR, 'heatmaps',
            f'data_source_{data_source.name}',
            f'subject_{data_source.ascended_being}',
            f'window_length_{self.window_length:0.2f}',
        )

        self.image_lists = {}
        for interp_type in Interpolation:
            data_dir = os.path.join(self.base_dir, f'interpolation_{interp_type.name}')
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)
            self.image_lists[interp_type.name] = {"images": [], "data_dir": data_dir}
        return

    def compute_images(self, spacing: float = 0.2):
        data_iter = list(self.data_source.window_generator(window_length=self.window_length, spacing=spacing))
        print(f'Using a single process to compute {len(data_iter)} images')
        pbar = tqdm(
            total=len(data_iter),
            desc=f'Computing images: length: {self.window_length:0.2f}, spacing: {spacing}',
            file=sys.stdout
        )
        # todo make multiprocess
        for window in data_iter:
            window_label_str = window['label'].value_counts().idxmax()
            window_idx = window['idx'].iloc[0]
            for interp_type in Interpolation:
                cv_image, interp_type = self.heatmap_from_window(window, interp_type)
                self.image_lists[interp_type.name]["images"].append({
                    "idx": window_idx, "event": window_label_str, "image": cv_image
                })
            pbar.update(1)
        pbar.close()
        return

    def heatmap_from_window(self, window, interp_type):
        window_data = window[self.data_source.coi]
        window_np_data = window_data.to_numpy()

        window_extrapolation = np.zeros((self.num_rows, 0), dtype='float32')
        for data_vals in window_np_data:
            col_vals = interpolate_row(data_vals, num_rows=self.num_rows, interp_type=interp_type)
            norm_vals = (col_vals - np.min(col_vals)) / np.ptp(col_vals)
            window_extrapolation = np.column_stack([window_extrapolation, norm_vals])

        color_image = self.cmap(window_extrapolation)[:, :, :-1]
        img = cv2.convertScaleAbs(color_image, alpha=255.0)
        cv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv_image, interp_type

    def save_heatmaps(self):
        for interp_type, image_info in self.image_lists.items():
            self.save_image_list(interp_type, image_info)
        return

    @staticmethod
    def save_image_list(interp_type, image_info):
        image_entries = image_info["images"]
        num_images = len(image_entries)
        pbar = tqdm(total=num_images, desc=f'Saving images: {interp_type}', file=sys.stdout)
        for entry in image_entries:
            event_type = entry["event"]
            event_dir = os.path.join(image_info["data_dir"], f'event_{event_type}')
            if not os.path.isdir(event_dir):
                os.makedirs(event_dir)

            img = entry["image"]
            img_fname = os.path.join(event_dir, f'idx_{entry["idx"]}_id_{img_id(img)}.png')
            cv2.imwrite(filename=img_fname, img=img)
            pbar.update(1)
        pbar.close()
        return


def main():
    trial_type = 'motor_imagery_right_left'
    num_subjects = -1
    subject_name = 'flat'
    save_method = 'csv'

    data_source = PhysioDataSource(subject=None, trial_type=trial_type, save_method=save_method)
    # data_source = RecordedDataSource(subject=subject_name, trial_type=trial_type)
    subject_list = data_source.subject_names
    if num_subjects > 0:
        subject_list = subject_list[:num_subjects]
    window_length_list = [0.2 * idx for idx in range(1, 4)]  # length in seconds
    num_rows = 200

    abs_s_time = time.time()
    for window_length in window_length_list:
        for subject_name in subject_list:
            print(f'Subject: {subject_name}\nWindow length: {window_length:0.2f}')
            data_source.set_subject(subject_name)
            data_transformer = DataTransformer(data_source, window_length=window_length, num_rows=num_rows)

            s_time = time.time()
            data_transformer.compute_images()
            e_time = time.time()
            d_time = e_time - s_time
            print(f'Time to compute images: {d_time:0.4f} seconds')

            s_time = time.time()
            data_transformer.save_heatmaps()
            e_time = time.time()
            d_time = e_time - s_time
            print(f'Time to save images: {d_time:0.4f} seconds')
            print('-----------------------------------------------------------------')
    abs_e_time = time.time()
    abs_d_time = abs_e_time - abs_s_time
    print(f'Time to iterate over all subjects and windows: {abs_d_time:0.4f} seconds')
    return


if __name__ == '__main__':
    main()
