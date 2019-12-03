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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from scipy.interpolate import interp1d
from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource
from utils.Misc import find_files_by_name


class CMAP(Enum):
    YlGnBu = auto()
    YlOrRd = auto()
    Reds = auto()
    rocket_r = auto()
    Spectral = auto()
    Blues = auto()
    BuGn = auto()
    BuPu = auto()
    Greens = auto()
    OrRd = auto()
    Oranges = auto()
    PuBu = auto()
    gist_heat = auto()
    viridis = auto()
    plasma = auto()
    inferno = auto()
    magma = auto()
    cividis = auto()
    copper = auto()


class Interpolation(Enum):
    LINEAR = 0
    QUADRATIC = 1
    CUBIC = 2


def interpolate_row(row_sample: np.ndarray, num_rows: int, interp_type: Interpolation):
    interp_str = interp_type.name.lower()
    # add a zero before interpolation to lock either end to ground state
    row_sample = np.insert(row_sample, 0, values=[0])
    row_sample = np.append(row_sample, 0)

    x = np.linspace(0, row_sample.shape[0], num=row_sample.shape[0], endpoint=False)
    xnew = np.linspace(0, row_sample.shape[0] - 1, num=num_rows, endpoint=True)

    interp_func = interp1d(x, row_sample, fill_value='extrapolate', kind=interp_str)
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


def sort_windows(window_entry):
    window_len = float(window_entry[0]) * 100
    return window_len


def plot_timing_boxplot(data_dict: dict, fig_title: str, x_title: str, y_title: str, subject_name: str,
                        fig_width: int = 12, fig_height: int = 12):
    style.use('ggplot')
    data_dict = sorted(data_dict.items(), key=sort_windows)
    x_labels = [entry[0] for entry in data_dict]
    y_vals = [entry[1] for entry in data_dict]

    fig, axes = plt.subplots(figsize=(fig_width, fig_height))
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
    axes.set_ylim(bottom=0, top=0.12)

    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()

    axes.set_title(fig_title, size=24)
    axes.set_xlabel(x_title, size=24)
    axes.set_ylabel(y_title, size=24)

    fig.tight_layout()

    save_dir = os.path.join(DATA_DIR, 'timing_plots', subject_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plot_fname = os.path.join(save_dir, f"{fig_title}.png")

    plot_fname = plot_fname.replace(' ', '_')
    plot_fname = plot_fname.replace(':', '_')

    plt.savefig(plot_fname)
    plt.close()
    return


def plot_size_scatter(data_dict: dict, fig_title: str, x_title: str, y_title: str,
                      fig_width: int = 12, fig_height: int = 12):
    style.use('ggplot')
    data_dict = sorted(data_dict.items(), key=sort_windows)
    x_labels = [entry[0] for entry in data_dict]
    y_vals = [entry[1] for entry in data_dict]

    fig, axes = plt.subplots(figsize=(fig_width, fig_height))
    axes.plot(x_labels, y_vals, 'g')
    axes.scatter(x_labels, y_vals, marker='o', color='b')

    axes.set_axisbelow(True)
    axes.minorticks_on()

    axes.grid(which='major', linestyle='-', linewidth='0.5', color='red')

    axes.tick_params(axis='y', labelcolor='k', labelsize=18)
    axes.tick_params(which='both', top='off', left='off', right='off', bottom='off')

    axes.set_xticklabels(x_labels, rotation=90, size=18)
    axes.set_title(fig_title, size=24)
    axes.set_xlabel(x_title, size=24)
    axes.set_ylabel(y_title, size=24)

    fig.tight_layout()
    save_dir = os.path.join(DATA_DIR, 'dataset_size_plots')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plot_fname = os.path.join(save_dir, f"{fig_title}.png")

    plot_fname = plot_fname.replace(' ', '_')
    plot_fname = plot_fname.replace(':', '_')

    plt.savefig(plot_fname)
    plt.close()
    return


class DataTransformer:

    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        self.cmap = matplotlib.cm.get_cmap(CMAP.gist_heat.name)

        self.subject_dir = None
        self.window_dir = None

        self.window_length = None
        self.num_rows = None
        self.num_cols = None
        return

    def build_window_dir(self):
        window_dir = os.path.join(
            self.subject_dir,
            f'window_length_{self.window_length:0.2f}',
        )
        if not os.path.isdir(window_dir):
            os.makedirs(window_dir)
        return window_dir

    def build_subject_dir(self):
        subject_dir = os.path.join(
            DATA_DIR, 'heatmaps',
            f'data_source_{self.data_source.name}',
            f'subject_{self.data_source.ascended_being}'
        )
        if not os.path.isdir(subject_dir):
            os.makedirs(subject_dir)
        return subject_dir

    def set_subject(self, subject_name):
        self.data_source.set_subject(subject_name)
        self.subject_dir = self.build_subject_dir()
        if self.window_length:
            self.window_dir = self.build_window_dir()
        return

    def set_window_length(self, window_length):
        self.window_length = window_length
        self.num_cols = int(self.data_source.sample_freq * window_length)
        if self.subject_dir:
            self.window_dir = self.build_window_dir()
        return

    def set_num_rows(self, num_rows):
        self.num_rows = num_rows
        return

    def _verify_properties(self):
        if not self.window_length:
            raise ValueError(f'Property not set: window_length\n\tHave you called set_window_length?')
        if not self.subject_dir:
            raise ValueError(f'Property not set: subject_dir\n\tHave you called set_subject?')
        if not self.window_dir:
            raise ValueError(f'Property not set: window_dir\n\tHave you called set_subject?')
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
        time_dict = {'image_count': 0}
        for interp_type in Interpolation:
            time_dict[interp_type.name] = []
        for window in data_iter:
            window_label_str = window['label'].value_counts().idxmax()
            window_idx = window['idx'].iloc[0]
            for interp_type in Interpolation:
                img_id = build_img_id(self.window_dir, window_label_str)
                heatmap_fname = build_heatmap_fname(
                    self.window_dir, interp_type.name, img_id, window_idx, window_label_str
                )
                if not os.path.isfile(heatmap_fname):
                    start_time = time.time()
                    cv_image = self.heatmap_from_window(window, interp_type)
                    end_time = time.time()
                    delta_time = end_time - start_time
                    time_dict[interp_type.name].append(delta_time)
                    time_dict['image_count'] += 1
                    img_entry = {"idx": window_idx, "event": window_label_str, "image": cv_image}
                    img_lists[interp_type.name].append(img_entry)
                    if save_image:
                        save_heatmap(cv_image, heatmap_fname)
            pbar.update(1)
        pbar.close()

        time_metric_fname = os.path.join(self.window_dir, f'heatmap_timings.json')
        if os.path.isfile(time_metric_fname):
            with open(time_metric_fname, 'r+') as timing_file:
                timing_info = json.load(timing_file)
            time_dict['image_count'] += timing_info['image_count']
            for interp_type in Interpolation:
                img_timings = timing_info[interp_type.name]
                time_dict[interp_type.name].extend(img_timings)
        with open(time_metric_fname, 'w+') as timing_file:
            json.dump(time_dict, timing_file, indent=2)
        return img_lists

    def plot_subject_timings(self):
        timing_fname_list = find_files_by_name('heatmap_timings', self.subject_dir)
        timing_windows_dict = {
            interp_type.name: {}
            for interp_type in Interpolation
        }
        for timing_fname in timing_fname_list:
            with open(timing_fname, 'r+') as timing_file:
                timing_dict = json.load(timing_file)
            window_length = os.path.basename(os.path.dirname(timing_fname)).split('_')[-1]
            for interp_type in Interpolation:
                timing_windows_dict[interp_type.name][window_length] = timing_dict[interp_type.name]

        for interp_type in Interpolation:
            plot_title = f'Time to compute {interp_type.name.lower()} signal images: {self.data_source.ascended_being}'
            plot_timing_boxplot(
                timing_windows_dict[interp_type.name],
                fig_title=plot_title,
                x_title='Window length (s)',
                y_title='Time per image (s)',
                subject_name=self.data_source.ascended_being
            )
        return

    def plot_dataset_size(self):
        timing_fname_list = find_files_by_name('heatmap_timings', self.subject_dir)
        window_count_dict = {}
        for timing_fname in timing_fname_list:
            with open(timing_fname, 'r+') as timing_file:
                timing_dict = json.load(timing_file)
            window_length = os.path.basename(os.path.dirname(timing_fname)).split('_')[-1]
            window_count_dict[window_length] = timing_dict['image_count']

        plot_title = f'Number of signal images: {self.data_source.ascended_being}'
        plot_size_scatter(
            window_count_dict,
            fig_title=plot_title,
            x_title='Window length (s)',
            y_title='Number of images (s)'
        )
        return

    def heatmap_from_window(self, window, interp_type):
        style.use('ggplot')
        matplotlib.use('Qt5Agg')
        window_data = window[self.data_source.coi]
        window_np_data = window_data.to_numpy()

        window_extrapolation = np.zeros((self.num_rows, 0), dtype='float32')
        for data_vals in window_np_data:
            data_vals = data_vals * 1E6
            col_vals = interpolate_row(data_vals, num_rows=self.num_rows, interp_type=interp_type)
            norm_vals = (col_vals - np.min(col_vals)) / np.ptp(col_vals)
            window_extrapolation = np.column_stack([window_extrapolation, norm_vals])

        color_image = self.cmap(window_extrapolation)[:, :, :-1]
        img = cv2.convertScaleAbs(color_image, alpha=255.0)
        cv_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv_image


def main():
    compute_images = True
    plot_timings = True
    ############################################
    trial_type = 'motor_imagery_right_left'
    num_subjects = -1
    subject_name = 'flat'
    save_method = 'h5'
    ############################################

    data_source = PhysioDataSource(subject=None, trial_type=trial_type, save_method=save_method)
    # data_source = RecordedDataSource(subject=subject_name, trial_type=trial_type)
    subject_list = sorted(data_source.subject_names)
    if num_subjects > 0:
        subject_list = subject_list[:num_subjects]
    window_length_list = [0.2 * idx for idx in range(1, 6)]  # length in seconds
    num_rows = 200

    data_transformer = DataTransformer(data_source)
    data_transformer.set_num_rows(num_rows)

    if compute_images:
        abs_s_time = time.time()
        for subject_name in subject_list:
            data_transformer.set_subject(subject_name)
            for window_length in window_length_list:
                data_transformer.set_window_length(window_length)

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

    if plot_timings:
        for subject_name in tqdm(subject_list):
            data_transformer.set_subject(subject_name)
            data_transformer.plot_subject_timings()
            data_transformer.plot_dataset_size()
    return


if __name__ == '__main__':
    main()
