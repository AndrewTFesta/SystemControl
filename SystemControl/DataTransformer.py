"""
@title
@description
"""
import hashlib
import time
from enum import Enum, auto

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import style, animation
from scipy.interpolate import interp1d

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


class DataTransformer:

    def __init__(self, data_source: DataSource, window_length: float, num_rows: int = 100, ):
        self.data_source = data_source
        self.window_length = window_length

        self._update_delay = 1. / self.data_source.sample_freq
        self.cmap = matplotlib.cm.get_cmap(CMAP.gist_heat.name)

        # image array setup and data
        self.num_rows = num_rows
        self.num_cols = int(data_source.sample_freq * window_length)
        self._linear_im_data = np.zeros((self.num_rows, self.num_cols), dtype='float32')
        self._quad_im_data = np.zeros((self.num_rows, self.num_cols), dtype='float32')
        self._cubic_im_data = np.zeros((self.num_rows, self.num_cols), dtype='float32')

        # figure settings
        ###################################################
        self._fig = plt.figure(facecolor='white')
        self._fig.subplots_adjust(
            # left=0.1,  # the left side of the subplots of the figure
            # right=0.7,  # the right side of the subplots of the figure
            # bottom=0.1,  # the bottom of the subplots of the figure
            # top=0.9,  # the top of the subplots of the figure
            # wspace=0.9,  # the amount of width reserved for blank space between subplots
            hspace=0.9,  # the amount of height reserved for white space between subplots
        )

        self._axes_text = self._fig.add_subplot(6, 1, 1)
        self._axes_linear = self._fig.add_subplot(4, 1, 2)
        self._axes_quad = self._fig.add_subplot(4, 1, 3)
        self._axes_cubic = self._fig.add_subplot(4, 1, 4)

        self._axes_idx_text_artist = self._axes_text.text(
            x=0.5, y=0.5, s='Sample time: 0',
            ha="center", va="top"
        )
        self._axes_text.set_facecolor('white')
        self._axes_text.set_xticks([])
        self._axes_text.set_yticks([])

        self._linear_img_artist = self._axes_linear.imshow(self._linear_im_data)
        self._axes_linear.set_xlabel('Linear interpolation')
        self._axes_linear.set_xticks([])
        self._axes_linear.set_yticks([])
        # x0, x1 = self._axes_linear.get_xlim()
        # y0, y1 = self._axes_linear.get_ylim()
        # self._axes_linear.set_aspect(abs(x1 - x0) / abs(y1 - y0))

        self._quad_img_artist = self._axes_quad.imshow(self._quad_im_data)
        self._axes_quad.set_xlabel('Quadratic interpolation')
        self._axes_quad.set_xticks([])
        self._axes_quad.set_yticks([])
        # x0, x1 = self._axes_quad.get_xlim()
        # y0, y1 = self._axes_quad.get_ylim()
        # self._axes_quad.set_aspect(abs(x1 - x0) / abs(y1 - y0))

        self._cub_img_artist = self._axes_cubic.imshow(self._cubic_im_data)
        self._axes_cubic.set_xlabel('Cubic interpolation')
        self._axes_cubic.set_xticks([])
        self._axes_cubic.set_yticks([])
        # x0, x1 = self._axes_cubic.get_xlim()
        # y0, y1 = self._axes_cubic.get_ylim()
        # self._axes_cubic.set_aspect(abs(x1 - x0) / abs(y1 - y0))

        self._ani = animation.FuncAnimation(
            self._fig, self.update_heatmap, self.data_source, interval=self._update_delay, blit=True, repeat=False
        )
        ###################################################
        plt.show()
        return

    def update_heatmap(self, update_args):
        sample_entry = update_args[0]
        sample_idx = sample_entry["idx"]
        sample_data = sample_entry["data"]
        data_vals = np.array(list(sample_data.values()))
        changed_artists = []

        self._axes_idx_text_artist.set_text(f'Sample number: {sample_idx}')
        changed_artists.append(self._axes_idx_text_artist)
        
        lin_col_vals = interpolate_row(data_vals, num_rows=self.num_rows, interp_type=Interpolation.LINEAR)
        lin_norm_vals = (lin_col_vals - np.min(lin_col_vals)) / np.ptp(lin_col_vals)
        self._linear_im_data = np.column_stack([self._linear_im_data, lin_norm_vals])
        self._linear_im_data = self._linear_im_data[:, 1:]
        linear_color_image = self.cmap(self._linear_im_data)[:, :, :-1]
        self._linear_img_artist.set_data(linear_color_image)
        changed_artists.append(self._linear_img_artist)

        quad_col_vals = interpolate_row(data_vals, num_rows=self.num_rows, interp_type=Interpolation.QUADRATIC)
        quad_norm_vals = (quad_col_vals - np.min(quad_col_vals)) / np.ptp(quad_col_vals)
        self._quad_im_data = np.column_stack([self._quad_im_data, quad_norm_vals])
        self._quad_im_data = self._quad_im_data[:, 1:]
        quad_color_image = self.cmap(self._quad_im_data)[:, :, :-1]
        self._quad_img_artist.set_data(quad_color_image)
        changed_artists.append(self._quad_img_artist)

        cub_col_vals = interpolate_row(data_vals, num_rows=self.num_rows, interp_type=Interpolation.CUBIC)
        cub_norm_vals = (cub_col_vals - np.min(cub_col_vals)) / np.ptp(cub_col_vals)
        self._cubic_im_data = np.column_stack([self._cubic_im_data, cub_norm_vals])
        self._cubic_im_data = self._cubic_im_data[:, 1:]
        cubic_color_image = self.cmap(self._cubic_im_data)[:, :, :-1]
        self._cub_img_artist.set_data(cubic_color_image)
        changed_artists.append(self._cub_img_artist)

        return changed_artists


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


def main():
    subject_name = 'S001'
    trial_type = 'motor_imagery_right_left'
    window_length = 0.2  # length in seconds
    num_rows = 200

    data_source = PhysioDataSource(subject=subject_name, trial_type=trial_type)
    s_time = time.time()
    data_transformer = DataTransformer(data_source, window_length=window_length, num_rows=num_rows)
    e_time = time.time()
    d_time = e_time - s_time
    print(f'Time to iterate over entire trial: {d_time:0.4f}')
    return


if __name__ == '__main__':
    main()
