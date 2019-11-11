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


def img_id(data_entry: np.ndarray):
    entry_hash = hashlib.sha3_256(data_entry.tobytes()).hexdigest()
    return entry_hash


class DataTransformer:

    def __init__(self, data_source: DataSource, window_length: float, num_rows: int = 100):
        self.data_source = data_source
        self.window_length = window_length
        self.num_samples = self.data_source.get_num_samples()
        self.cmap = matplotlib.cm.get_cmap(CMAP.gist_heat.name)
        self.num_rows = num_rows
        self.num_cols = int(data_source.sample_freq * window_length)

        self.base_dir = os.path.join(
            DATA_DIR, 'heatmaps', data_source.name,
            f'window_length_{window_length:0.2f}', f'subject_{data_source.ascended_being}'
        )

        self.image_lists = {}
        for interp_type in Interpolation:
            data_dir = os.path.join(self.base_dir, interp_type.name)
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir)
            self.image_lists[interp_type.name] = {"images": [], "data_dir": data_dir}
        # self.linear_images = []
        # self.quadratic_images = []
        # self.cubic_images = []

        # self.linear_data_dir = os.path.join(self.base_dir, 'linear')
        # if not os.path.isdir(self.linear_data_dir):
        #     os.makedirs(self.linear_data_dir)
        #
        # self.quadratic_data_dir = os.path.join(self.base_dir, 'quadratic')
        # if not os.path.isdir(self.quadratic_data_dir):
        #     os.makedirs(self.quadratic_data_dir)
        #
        # self.cubic_data_dir = os.path.join(self.base_dir, 'cubic')
        # if not os.path.isdir(self.cubic_data_dir):
        #     os.makedirs(self.cubic_data_dir)

        # image array setup and data
        # self._linear_im_data = np.zeros((self.num_rows, self.num_cols), dtype='float32')
        # self._quad_im_data = np.zeros((self.num_rows, self.num_cols), dtype='float32')
        # self._cubic_im_data = np.zeros((self.num_rows, self.num_cols), dtype='float32')
        return

    def compute_images(self, spacing: float = 0.2):
        data_iter = self.data_source.window_generator(window_length=self.window_length, spacing=spacing)
        pbar = tqdm(
            desc=f'Computing images: length: {self.window_length}, spacing: {spacing}',
            file=sys.stdout
        )
        for entry in data_iter:
            sample_list = entry[0]
            event_entry = entry[1]
            event_type = event_entry["event_type"]
            sample_idx = sample_list[0]["idx"]

            lin_window_data = np.zeros((self.num_rows, 0), dtype='float32')
            quad_window_data = np.zeros((self.num_rows, 0), dtype='float32')
            cubic_window_data = np.zeros((self.num_rows, 0), dtype='float32')
            for sample_entry in sample_list:
                sample_data = sample_entry["data"]
                data_vals = np.array(list(sample_data.values()))

                #####
                lin_col_vals = interpolate_row(data_vals, num_rows=self.num_rows, interp_type=Interpolation.LINEAR)
                lin_norm_vals = (lin_col_vals - np.min(lin_col_vals)) / np.ptp(lin_col_vals)
                lin_window_data = np.column_stack([lin_window_data, lin_norm_vals])
                #####
                quad_col_vals = interpolate_row(data_vals, num_rows=self.num_rows, interp_type=Interpolation.QUADRATIC)
                quad_norm_vals = (quad_col_vals - np.min(quad_col_vals)) / np.ptp(quad_col_vals)
                quad_window_data = np.column_stack([quad_window_data, quad_norm_vals])
                #####
                cubic_col_vals = interpolate_row(data_vals, num_rows=self.num_rows, interp_type=Interpolation.CUBIC)
                cubic_norm_vals = (cubic_col_vals - np.min(cubic_col_vals)) / np.ptp(cubic_col_vals)
                cubic_window_data = np.column_stack([cubic_window_data, cubic_norm_vals])
                #####
            #####
            linear_color_image = self.cmap(lin_window_data)[:, :, :-1]
            lin_img = cv2.convertScaleAbs(linear_color_image, alpha=255.0)
            lin_cv_image = cv2.cvtColor(lin_img, cv2.COLOR_RGB2BGR)
            self.linear_images.append({"idx": sample_idx, "event": event_type, "image": lin_cv_image})
            #####
            quad_color_image = self.cmap(quad_window_data)[:, :, :-1]
            quad_img = cv2.convertScaleAbs(quad_color_image, alpha=255.0)
            quad_cv_image = cv2.cvtColor(quad_img, cv2.COLOR_RGB2BGR)
            self.quadratic_images.append({"idx": sample_idx, "event": event_type, "image": quad_cv_image})
            #####
            cubic_color_image = self.cmap(cubic_window_data)[:, :, :-1]
            cubic_img = cv2.convertScaleAbs(cubic_color_image, alpha=255.0)
            cubic_cv_image = cv2.cvtColor(cubic_img, cv2.COLOR_RGB2BGR)
            self.cubic_images.append({"idx": sample_idx, "event": event_type, "image": cubic_cv_image})
            #####
            pbar.update(1)
        pbar.close()
        return

    def save_heatmaps(self):
        self.save_image_list(self.linear_images, self.linear_data_dir, 'linear')
        self.save_image_list(self.quadratic_images, self.quadratic_data_dir, 'quadratic')
        self.save_image_list(self.cubic_images, self.cubic_data_dir, 'cubic')
        return

    @staticmethod
    def save_image_list(img_entries, base_dir, img_type):
        pbar = tqdm(total=len(img_entries), desc=f'Saving images: {img_type}', file=sys.stdout)
        for entry in img_entries:
            event_type = entry["event"]
            event_dir = os.path.join(base_dir, f'{event_type}')
            if not os.path.isdir(event_dir):
                os.makedirs(event_dir)

            img = entry["image"]
            img_idx = entry["idx"]
            # img = cv2.convertScaleAbs(img, alpha=255.0)
            # cv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_fname = os.path.join(event_dir, f'{img_type}_{img_idx}_{img_id(img)}.png')
            cv2.imwrite(filename=img_fname, img=img)
            pbar.update(1)
        pbar.close()
        return


def main():
    trial_type = 'motor_imagery_right_left'
    data_source = PhysioDataSource(subject=None, trial_type=trial_type)
    num_subjects = 1
    subject_list = data_source.subject_names[:num_subjects]
    del data_source
    # window_length_list = [0.2 * idx for idx in range(1, 4)]  # length in seconds
    window_length_list = [0.2]  # length in seconds
    num_rows = 200
    spacing = 0.2

    abs_s_time = time.time()
    for window_length in window_length_list:
        for subject_name in subject_list:
            data_source = PhysioDataSource(subject=subject_name, trial_type=trial_type)
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

            print(f'Subject: {subject_name}\nWindow length: {window_length}')
            print(f'Number of linear images: {len(data_transformer.linear_images)}')
            print(f'Number of quadratic images: {len(data_transformer.quadratic_images)}')
            print(f'Number of cubic images: {len(data_transformer.cubic_images)}')
            print('-----------------------------------------------------------------')
    abs_e_time = time.time()
    abs_d_time = abs_e_time - abs_s_time
    print(f'Time to iterate over all subjects and windows: {abs_d_time:0.4f} seconds')
    return


if __name__ == '__main__':
    main()
