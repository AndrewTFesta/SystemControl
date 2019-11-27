"""
@title
@description
"""
import hashlib
import json
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


def build_img_hash(data_entry: np.ndarray):
    entry_hash = hashlib.sha3_256(data_entry.tobytes()).hexdigest()
    return entry_hash


def build_img_id(base_dir: str, window_label: str):
    img_str = f'{base_dir}:{window_label}'
    entry_hash = hashlib.sha3_256(img_str.encode('utf-8')).hexdigest()
    return entry_hash


def save_heatmap(image, image_fname):
    event_dir = os.path.dirname(image_fname)
    if not os.path.isdir(event_dir):
        os.makedirs(event_dir)

    cv2.imwrite(filename=image_fname, img=image)
    return


def build_heatmap_fname(base_dir, inter_type, img_id, img_idx, img_event):
    event_dir = os.path.join(base_dir, f'interpolation_{inter_type}', f'event_{img_event}')
    img_fname = os.path.join(event_dir, f'idx_{img_idx}_id_{img_id}.png')
    return img_fname


class DataTransformer:

    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        self.cmap = matplotlib.cm.get_cmap(CMAP.gist_heat.name)

        self.window_length = None
        self.base_dir = None
        self.num_rows = None
        self.num_cols = None
        return

    def build_base_dir(self):
        base_dir = os.path.join(
            DATA_DIR, 'heatmaps',
            f'data_source_{self.data_source.name}',
            f'subject_{self.data_source.ascended_being}',
            f'window_length_{self.window_length:0.2f}',
        )
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        return base_dir

    def set_subject(self, subject_name):
        self.data_source.set_subject(subject_name)
        return

    def set_window_length(self, window_length):
        self.window_length = window_length
        self.num_cols = int(self.data_source.sample_freq * window_length)
        return

    def set_num_rows(self, num_rows):
        self.num_rows = num_rows
        return

    def set_base_dir(self):
        self.base_dir = self.build_base_dir()
        return

    def _verify_properties(self):
        if not self.window_length:
            raise ValueError(f'Property not set: window_length\n\tHave you called set_window_length?')
        if not self.base_dir:
            raise ValueError(f'Property not set: base_dir\n\tHave you called set_base_dir?')
        if not self.num_rows:
            raise ValueError(f'Property not set: num_rows\n\tHave you called set_num_rows?')
        return

    def compute_images(self, window_overlap: float = 0.2, save_image: bool = False):
        self._verify_properties()

        data_iter = list(
            self.data_source.window_generator(window_length=self.window_length, window_overlap=window_overlap)
        )
        img_lists = {
            interp_type.name: []
            for interp_type in Interpolation
        }

        print(f'Using a single process to compute {len(data_iter)} images')
        pbar = tqdm(
            total=len(data_iter),
            desc=f'Computing images: length: {self.window_length:0.2f}, spacing: {window_overlap}',
            file=sys.stdout
        )
        # todo make multiprocess
        time_dict = {'time_per_image': [], 'image_count': 0}
        for window in data_iter:
            window_label_str = window['label'].value_counts().idxmax()
            window_idx = window['idx'].iloc[0]
            for interp_type in Interpolation:
                img_id = build_img_id(self.base_dir, window_label_str)
                heatmap_fname = build_heatmap_fname(
                    self.base_dir, interp_type.name, img_id, window_idx, window_label_str
                )
                if not os.path.isfile(heatmap_fname):
                    start_time = time.time()
                    cv_image = self.heatmap_from_window(window, interp_type)
                    end_time = time.time()
                    delta_time = end_time - start_time
                    time_dict['time_per_image'].append(delta_time)
                    time_dict['image_count'] += 1
                    img_entry = {"idx": window_idx, "event": window_label_str, "image": cv_image}
                    img_lists[interp_type.name].append(img_entry)
                    if save_image:
                        save_heatmap(cv_image, heatmap_fname)
            pbar.update(1)
        pbar.close()

        time_metric_fname = os.path.join(self.base_dir, f'heatmap_timings.json')
        with open(time_metric_fname, 'w+') as metrics_file:
            json.dump(time_dict, metrics_file)
        return img_lists

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
        return cv_image


def main():
    trial_type = 'motor_imagery_right_left'
    num_subjects = -1
    subject_name = 'flat'
    save_method = 'csv'

    data_source = PhysioDataSource(subject=None, trial_type=trial_type, save_method=save_method)
    # data_source = RecordedDataSource(subject=subject_name, trial_type=trial_type)
    subject_list = sorted(data_source.subject_names)
    if num_subjects > 0:
        subject_list = subject_list[:num_subjects]
    window_length_list = [0.2 * idx for idx in range(1, 6)]  # length in seconds
    num_rows = 200

    data_transformer = DataTransformer(data_source)
    data_transformer.set_num_rows(num_rows)

    abs_s_time = time.time()
    for subject_name in subject_list:
        data_transformer.set_subject(subject_name)
        for window_length in window_length_list:
            data_transformer.set_window_length(window_length)
            data_transformer.set_base_dir()

            print('-----------------------------------------------------------------')
            print(f'Subject: {subject_name}\nWindow length: {window_length:0.2f}')
            s_time = time.time()
            data_transformer.compute_images(save_image=True)
            e_time = time.time()
            d_time = e_time - s_time
            print(f'Time to compute and save images: {d_time:0.4f} seconds')
            print('-----------------------------------------------------------------')
    abs_e_time = time.time()
    abs_d_time = abs_e_time - abs_s_time
    print(f'Time to iterate over all subjects and windows: {abs_d_time:0.4f} seconds')
    return


if __name__ == '__main__':
    main()
