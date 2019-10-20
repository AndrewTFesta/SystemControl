"""
@title
@description

    |---------------------------------------------------|
    | run         | task                                |
    |---------------------------------------------------|
    | 1           | Baseline, eyes open                 |
    | 2           | Baseline, eyes closed               |
    | 3, 7, 11    | Motor execution: left vs right hand |
    | 4, 8, 12    | Motor imagery: left vs right hand   |
    | 5, 9, 13    | Motor execution: hands vs feet      |
    | 6, 10, 14   | Motor imagery: hands vs feet        |
    |---------------------------------------------------|
    To download any of the datasets, use the data_path (fetches full dataset)
    or the load_data (fetches dataset partially) functions.
"""
import os
import shutil
import sys
import time
import urllib.parse
import zipfile

import numpy as np
import pyedflib

import requests
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from requests.exceptions import HTTPError
from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.stacklineplot import stackplot
from SystemControl.utilities import find_files_by_type, time_function

DATASET_NAME = 'eeg-motor-movementimagery-dataset-1.0.0'
EXTERNAL_ZIP_URL = urllib.parse.urljoin(
    'https://physionet.org/static/published-projects/eegmmidb/',
    '{}.zip'.format(DATASET_NAME)
)
DATASET_LOCATION = os.path.join(DATA_DIR, DATASET_NAME)
LOCAL_ZIP_PATH = os.path.join(DATASET_LOCATION, '{}.zip'.format(DATASET_NAME))


def download_dataset_zip(force_download=True) -> bool:
    # remove existing and re-download
    if os.path.exists(LOCAL_ZIP_PATH):
        print('Zip file already located in default location:\n{}'.format(LOCAL_ZIP_PATH))
        if not force_download:
            print('Not re-downloading zip file')
            return False
        else:
            print('Removing previously downloaded zip file')
            os.remove(LOCAL_ZIP_PATH)

    # TODO add ability to specify custom location
    if not os.path.exists(DATASET_LOCATION):
        os.mkdir(DATASET_LOCATION)

    download_success = False
    try:
        print('Starting zip download...')
        response = requests.get(EXTERNAL_ZIP_URL, stream=True)
        headers = response.headers

        content_length = int(headers['content-length'])
        part_zip = os.path.join(DATASET_LOCATION, '{}.part_zip'.format(DATASET_NAME))
        print('Total size of zip file: {} bytes'.format(content_length))

        unit = 'B'
        unit_scale = True
        c_size = 512
        pbar_format = 'Downloaded: {percentage:.4f}% {r_bar}'
        download_progress = tqdm(
            total=content_length,
            bar_format=pbar_format,
            unit=unit,
            unit_scale=unit_scale,
            leave=True,
            file=sys.stdout
        )
        with open(part_zip, 'wb+') as handle:
            for chunk in response.iter_content(chunk_size=c_size):
                if chunk:  # filter out keep-alive new chunks
                    handle.write(chunk)
                    download_progress.update(c_size)
        download_progress.close()
        print('Time to download: {:.4f} s'.format(download_progress.format_dict['elapsed']))
        os.rename(part_zip, LOCAL_ZIP_PATH)
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # Python 3.6
        download_success = False
    except Exception as err:
        print(f'Other error occurred: {err}')  # Python 3.6
        download_success = False
    return download_success


def unzip_data(remove_zip=False, force_unzip=True) -> bool:
    print('Unzipping zip file')
    unzip_dir = os.path.join(os.path.join(DATASET_LOCATION, DATASET_NAME))

    if os.path.exists(unzip_dir):
        print('Located unzipped directory')
        if not force_unzip:
            print('Not re-unzipping')
            return False
        print('Removing unzipped directory')
        shutil.rmtree(unzip_dir)
    with zipfile.ZipFile(LOCAL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    if remove_zip:
        os.remove(LOCAL_ZIP_PATH)  # remove zip file after extracting
        print('Removed temporary zip file')
    return True


def format_dataset_dir():
    unzip_dir = os.path.join(os.path.join(DATASET_LOCATION, DATASET_NAME))
    time_start = time.time()
    # extract useful data
    physio_files = find_files_by_type(file_type='edf', root_dir=unzip_dir)
    time_end = time.time()
    print('Time to find edf files: {:.4f}'.format(time_end - time_start))
    print('Found {} EDF files'.format(len(physio_files)))

    time_start = time.time()
    add_edf_range(physio_files)  # not implemented
    time_end = time.time()
    print('Time to store edf files: {:.4f}'.format(time_end - time_start))

    edf_data_list = []
    for each_file in physio_files:
        raw_edf = pyedflib.EdfReader(each_file)
        edf_data_list.append(raw_edf)
        raw_edf._close()
    print(len(physio_files))
    print(len(edf_data_list))
    return edf_data_list


def add_edf_range(edf_file_list):
    for each_edf_file in edf_file_list:
        add_edf(each_edf_file)
    return


def add_edf(edf_fname):
    if not os.path.isfile(edf_fname):
        print('Unable to locate file: {}'.format(edf_fname))
        return

    # TODO add to eeg db
    # TODO implement functionality
    return


def get_num_signals(raw_edf):
    return raw_edf.signals_in_file


def get_signal_labels(raw_edf):
    return raw_edf.getSignalLabels()


def get_signal_values(raw_edf):
    # TODO limit to 3 channels + reference
    sigs = get_num_signals(raw_edf)
    sigbufs = np.zeros((sigs, raw_edf.getNSamples()[0]))
    for i in np.arange(sigs):
        sigbufs[i, :] = raw_edf.readSignal(i)
    return sigbufs


def get_annotations(raw_edf):
    annotations = raw_edf.readAnnotations()
    for n in np.arange(raw_edf.annotations_in_file):
        print("annotation: onset is %f    duration is %s    description is %s" % (
            annotations[0][n], annotations[1][n], annotations[2][n]))
    return annotations


def get_n_samples(raw_edf, channel_num, num_samples):
    buf = raw_edf.readSignal(channel_num)
    result = ""
    for i in np.arange(num_samples):
        result += ("%.1f, " % buf[i])
    print(result)
    return


def plot_signals(edf_file):
    raw_edf = pyedflib.EdfReader(edf_file)
    n = raw_edf.signals_in_file
    signal_labels = raw_edf.getSignalLabels()
    n_min = raw_edf.getNSamples()[0]
    sigbufs = [np.zeros(raw_edf.getNSamples()[i]) for i in np.arange(n)]
    for i in np.arange(n):
        sigbufs[i] = raw_edf.readSignal(i)
        if n_min < len(sigbufs[i]):
            n_min = len(sigbufs[i])
    raw_edf._close()
    del raw_edf

    n_plot = np.min((n_min, 2000))
    sigbufs_plot = np.zeros((n, n_plot))
    for i in np.arange(n):
        sigbufs_plot[i, :] = sigbufs[i][:n_plot]

    stackplot(sigbufs_plot[:, :n_plot], ylabels=signal_labels)
    return


def save_dataset():
    return


def load_data(subject: int, runs: list, verbosity: str = 'critical') -> list:
    """
    Note:   Raises a RuntimeWarning:
                Limited 1 annotation(s) that were expanding outside the data range.
                Subject 100
                    Runs: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14

    :param subject:
    :param runs:
    :param verbosity:
    :return:
    """
    run_str = ','.join(map(str, runs))
    print('Loading dataset: subject: {}: runs: {}'.format(subject, run_str))
    subject_dir = os.path.join(os.path.join(DATASET_LOCATION, DATASET_NAME, 'files', 'S{:03d}'.format(subject)))
    if not os.path.isdir(subject_dir):
        print('No directory located for subject: {}'.format(subject))
        return []

    physio_files = [
        os.path.join(subject_dir, 'S{:03d}R{:02d}.edf'.format(subject, each_run))
        for each_run in runs
        if os.path.isfile(os.path.join(subject_dir, 'S{:03d}R{:02d}.edf'.format(subject, each_run)))
    ]

    edf_data_list = []
    for each_file in physio_files:
        raw_edf = pyedflib.EdfReader(each_file)
        # TODO extract useful info and close file
        edf_data_list.append(raw_edf)
        raw_edf._close()
    return edf_data_list

    ########################################

    edf_data_list = []
    for each_file in physio_files:
        raw_edf = read_raw_edf(each_file, preload=True, verbose=verbosity)
        edf_data_list.append(raw_edf)

    ########################################
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    raw = concatenate_raws([read_raw_edf(each_file, preload=True) for each_file in physio_files])
    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    ########################################

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
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

    ########################################

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
    return edf_data_list


# noinspection DuplicatedCode
def manual_download():
    force_download = False
    remove_zip = False
    force_unzip = False

    subject_range = list(range(1, 110))
    run_list = list(range(1, 15))
    subject = 1
    hf_runs = [6, 10, 14]  # motor imagery: hands vs feet

    _, time_to_run = time_function(download_dataset_zip, force_download=force_download)
    print('Time to run: {:.4f}'.format(time_to_run))

    _, time_to_run = time_function(unzip_data, remove_zip=remove_zip, force_unzip=force_unzip)
    print('Time to run: {:.4f}'.format(time_to_run))

    format_dataset_dir()
    physio_ds, time_to_run = time_function(format_dataset_dir)
    print('Time to run: {:.4f}'.format(time_to_run))

    physio_ds, time_to_run = time_function(load_data, subject, hf_runs)
    print('Time to run: {:.4f}'.format(time_to_run))

    # unzip_dir = os.path.join(os.path.join(DATASET_LOCATION, DATASET_NAME))
    # time_start = time.time()
    # # extract useful data
    # physio_files = find_files_by_type(file_type='edf', root_dir=unzip_dir)
    # time_end = time.time()
    # print('Time to find edf files: {:.4f}'.format(time_end - time_start))
    # print('Found {} EDF files'.format(len(physio_files)))
    #
    # for each_file in physio_files:
    #     plot_signals(each_file)
    return


def main():
    """

    :return:
    """
    # todo refactor to use h5py
    manual_download()
    return


if __name__ == '__main__':
    main()
