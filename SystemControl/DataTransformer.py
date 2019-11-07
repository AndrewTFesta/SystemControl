"""
@title
@description
"""
import hashlib
import json
import os
import shutil
import time
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from scipy.interpolate import interp1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource
from SystemControl.DataSource.RecordedDataSource import RecordedDataSource
from SystemControl.utilities import find_files_by_type


class CMAP(Enum):
    rocket_r = 0
    YlGnBu = 1


class Interpolation(Enum):
    LINEAR = 0
    QUADRATIC = 1
    CUBIC = 2


def build_heatmap(image_slice: np.ndarray, cmap=CMAP.rocket_r, normalize=True):
    # todo use funcanimation to speed up building and saving
    if normalize:
        image_slice = minmax_scale(image_slice, feature_range=(0, 255), axis=0, copy=True)
    heat_map = sns.heatmap(image_slice, xticklabels=False, yticklabels=False, cmap=cmap.name, cbar=False)
    sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)

    fig = heat_map.get_figure()
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image_from_plot


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


def slice2img(data_slice: np.ndarray, row_spacing: int, interpolation: Interpolation):
    # beginning of slice corresponds to values earlier in time
    # place one row between each and one at either end
    # add number of rows once more to account for each signal
    num_rows = (data_slice.shape[1] + 1) * row_spacing + data_slice.shape[1]

    spaced_slice = []
    for each_row in data_slice:
        interp_row = interpolate_row(each_row, num_rows=num_rows, interp_type=interpolation)
        spaced_slice.append(interp_row)
    np_spaced_slice = np.transpose(spaced_slice)
    return np_spaced_slice


def slice_data(trial_info, num_start_samples_padding, num_samples_after_event) -> dict:
    data_samples = trial_info.samples
    data_events = trial_info.events

    # event_boundaries = [each_event.idx for each_event in data_events]
    event_names = [each_event.type for each_event in data_events]
    slice_dict = {event_name: [] for event_name in np.unique(event_names)}

    for each_event in data_events:
        new_slice = extract_slice(data_samples, each_event.idx, num_start_samples_padding, num_samples_after_event)
        if new_slice:
            slice_dict[each_event.type].append(new_slice)
    return slice_dict


def extract_slice(data, event_onset: int, num_start_samples_padding: int, num_samples_after_event: int):
    start_sample_idx = event_onset - num_start_samples_padding
    end_sample_idx = event_onset + num_samples_after_event

    # set boundary conditions for start and end of slice
    if start_sample_idx < 0:
        start_sample_idx = 0
    if end_sample_idx >= len(data):
        end_sample_idx = -1

    data_slice = data[start_sample_idx:end_sample_idx]
    return data_slice


def img_id(data_entry: np.ndarray):
    entry_hash = hashlib.sha3_256(data_entry.tobytes()).hexdigest()
    return entry_hash


def generate_csp_dataset(data_source: DataSource, subject: str):
    mne.set_log_level('CRITICAL')  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
    # # Set parameters and read data
    # avoid classification of evoked responses by using epochs that start 1s after cue onset.
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    subject = 1  # todo check for each subject
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])

    # # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))
    # raw.pick_channels(['C3', 'Cz', 'C4'])  # fails when reducing number of channels

    # # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    # epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    # # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    layout = read_layout('EEG1005')
    csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg', units='Patterns (AU)', size=1.5)

    sfreq = raw.info['sfreq']
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for sigs in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, sigs:(sigs + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    plt.show()
    return


def generate_heatmap_dataset(
        data_source: DataSource, subject: str, start_padding: float, end_padding: float, spacing: int,
        interpolation: Interpolation, cmap: CMAP, timing_resolution: int, verbose: bool = True, save_dir: str = None
):
    if verbose:
        print('------------------------------------------')
        print(f'Generating dataset')
        print(f'\tsubject: {subject}, spad: {start_padding:0.2f}, epad: {end_padding:0.2f}')
        print(f'\tspacing: {spacing}, interp: {interpolation.name}, cmap: {cmap.name}, res: {timing_resolution}')
        print('------------------------------------------')
    freq = data_source.sample_freq
    num_start_samples_padding = int(freq * start_padding)
    num_samples_after_event = int(freq * end_padding)

    s_time = time.time()
    ds_trials = data_source.get_subject_entries()  # todo for RecordedDataSet
    data_slice_dict = {}
    for each_trial in ds_trials:
        trial_slices = slice_data(each_trial, num_start_samples_padding, num_samples_after_event)
        data_slice_dict[each_trial.trial] = trial_slices
    e_time = time.time()
    if verbose:
        print(f'Time to slice data: {e_time - s_time:0.4f} seconds')

    s_time = time.time()
    img_slice_dict = {
        each_event: []
        for each_event in data_source.event_names
    }
    for trial_name, event_dict in data_slice_dict.items():
        for event_name, event_slices in event_dict.items():
            for each_slice in event_slices:
                slice_signals = np.array([each_entry.data for each_entry in each_slice])
                slice_img = slice2img(slice_signals, row_spacing=spacing, interpolation=interpolation)
                img_slice_dict[event_name].append(slice_img)
    e_time = time.time()
    if verbose:
        print(f'Time to convert slices to images: {e_time - s_time:0.4f} seconds')

    s_time = time.time()
    heatmap_dict = {
        each_event: []
        for each_event in data_source.event_names
    }
    for img_event, img_slice_list in img_slice_dict.items():
        for img_slice in img_slice_list:
            heatmap_image = build_heatmap(img_slice, cmap=cmap)
            heatmap_dict[img_event].append(heatmap_image)
    e_time = time.time()
    if verbose:
        print(f'Time to build heatmaps: {e_time - s_time:0.4f} seconds')

    s_time = time.time()
    for heatmap_event, heatmap_list in heatmap_dict.items():
        if save_dir:
            base_dir = save_dir
        else:
            base_dir = os.path.join(
                DATA_DIR, 'heatmaps', data_source.name,
                f'spad_{int(start_padding * timing_resolution)}', f'epad_{int(end_padding * timing_resolution)}',
                f'{interpolation.name}', f'{subject}', f'{heatmap_event}'
            )

        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)
        for heatmap_idx, each_heatmap in enumerate(heatmap_list):
            out_fname = os.path.join(base_dir, f'{img_id(each_heatmap)}.png')
            if os.path.isfile(out_fname):
                os.remove(out_fname)
            # opencv faster than pillow: 2.63 it/s PIL vs 3.48 it/s CV2
            # opencv faster than pyvips: 10.0500 s pyvips vs 8.9690 s opencv2
            cv2.imwrite(out_fname, each_heatmap)
    e_time = time.time()
    if verbose:
        print(f'Time to save images: {e_time - s_time:0.4f} seconds')
        print('------------------------------------------')
        print(f'Generating dataset completed')
        print('------------------------------------------')
    return


def main():
    data_source = PhysioDataSource()
    # data_source = RecordedDataSource()

    timing_resolution = 100
    start_padding = int(0.1 * timing_resolution) / timing_resolution
    end_padding_range = [
        int(each_epad * timing_resolution) / timing_resolution
        for each_epad in [0.1, 0.2, 0.3, 0.4, 0.5]
    ]
    spacing = 100
    cmap = CMAP.rocket_r
    verbose = False

    total_count = len(data_source.subject_names)
    total_count *= len(Interpolation.__members__)
    total_count *= len(end_padding_range)

    pbar = tqdm(total=total_count, desc='Generating heatmap dataset')
    timing_dict = {}
    for each_subject in data_source.subject_names:
        data_source.set_subject(each_subject)
        timing_dict[each_subject] = {}
        for each_interpolation in Interpolation:
            timing_dict[each_subject][each_interpolation.name] = {}
            for each_end_padding in end_padding_range:
                save_dir = os.path.join(
                    DATA_DIR, 'heatmaps', data_source.name,
                    f'spad_{int(start_padding * timing_resolution)}',
                    f'epad_{int(each_end_padding * timing_resolution)}',
                    f'{each_interpolation.name}', f'{each_subject}'
                )
                s_time = time.time()
                generate_heatmap_dataset(
                    data_source=data_source, subject=each_subject, interpolation=each_interpolation,
                    end_padding=each_end_padding, start_padding=start_padding, spacing=spacing,
                    cmap=cmap, timing_resolution=timing_resolution, verbose=verbose, save_dir=save_dir
                )
                e_time = time.time()
                d_time = e_time - s_time
                num_images = len(find_files_by_type('png', save_dir))
                timing_dict[each_subject][each_interpolation.name][each_end_padding] = {
                    'time': d_time, 'num_images': num_images
                }
                # print(f'Time to build dataset: {each_subject}:{each_interpolation.name}:{each_end_padding}:'
                #       f'\n\tnumber of images: {num_images}, {d_time:0.4f} seconds')
                pbar.update(1)
    pbar.close()

    save_dir = os.path.join(DATA_DIR, 'timings')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    timing_fname = os.path.join(save_dir, f'generate_heatmap_dataset.json')
    with open(timing_fname, 'w+') as timing_file:
        json.dump(timing_dict, timing_file, indent=2)
    return


if __name__ == '__main__':
    main()
