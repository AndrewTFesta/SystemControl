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

    T0  rest
    T1  onset of motion (real or imagined)
            the left fist (in runs 3, 4, 7, 8, 11, and 12)
            both fists (in runs 5, 6, 9, 10, 13, and 14)
    T2  onset of motion (real or imagined)
            the right fist (in runs 3, 4, 7, 8, 11, and 12)
            both feet (in runs 5, 6, 9, 10, 13, and 14)
"""
import multiprocessing as mp
import os
import sys
import threading
import time
import urllib.parse

import mne
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from tqdm import tqdm

from SystemControl.DataSource.DataSource import DataSource, build_entry_id
from SystemControl.utilities import download_large_file, unzip_file, find_files_by_type, idx_to_time

DEBUG = False


def int_to_subject_str(subject_int):
    return f'S{subject_int:03d}'


def int_to_run_str(run_int):
    return f'R{run_int:02d}'


class PhysioDataSource(DataSource):

    def __init__(self, subject: str = None, trial_type: str = None, log_level: str = 'CRITICAL',
                 save_method: str = 'h5'):
        mne.set_log_level(log_level)  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
        DataSource.__init__(self, log_level=log_level, save_method=save_method)

        self.name = 'Physio'
        self.sample_freq = 160

        num_subjects = 109
        if DEBUG:
            num_subjects = 2
        self.subject_names = [int_to_subject_str(each_subject) for each_subject in list(range(1, num_subjects + 1))]

        self.__trial_mappings = {
            'baseline_open': [int_to_run_str(each_run) for each_run in [1]],
            'baseline_closed': [int_to_run_str(each_run) for each_run in [2]],
            'motor_execution_right_left': [int_to_run_str(each_run) for each_run in [3, 7, 11]],
            'motor_execution_hands_feet': [int_to_run_str(each_run) for each_run in [5, 9, 13]],
            'motor_imagery_right_left': [int_to_run_str(each_run) for each_run in [4, 8, 12]],
            'motor_imagery_hands_feet': [int_to_run_str(each_run) for each_run in [6, 10, 14]]
        }
        self.trial_types = list(self.__trial_mappings.keys())
        self.event_names = ['T0', 'T1', 'T2']
        self.ascended_being = subject if subject else self.subject_names[0]
        self.selected_trial_type = trial_type if trial_type else self.trial_types[4]

        self._validated = False
        if not self.__validate_dataset():
            self.__download_dataset(use_mp=True, debug=DEBUG)
            self.save_data()
        else:
            self.load_data()
        self.set_subject(self.ascended_being)
        return

    def __trial_type_from_name(self, trial_name):
        entry_trial_type = None
        for trial_type, trial_names in self.__trial_mappings.items():
            if trial_name in trial_names:
                entry_trial_type = trial_type
                break
        return entry_trial_type

    def __clean_raw_edf(self, raw_edf):
        cleaned_raw_edf = raw_edf.copy()
        cleaned_raw_edf.rename_channels(lambda x: x.strip('.'))
        cleaned_raw_edf.pick_channels(self.coi)
        cleaned_raw_edf = cleaned_raw_edf.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        return cleaned_raw_edf

    def __download_dataset(self, force_download=False, force_unzip=False, use_mp: bool = True, debug: bool = False):
        external_name = 'eeg-motor-movementimagery-dataset-1.0.0'
        external_zip_url = urllib.parse.urljoin(
            'https://physionet.org/static/published-projects/eegmmidb/',
            f'{external_name}.zip'
        )
        zip_name = download_large_file(
            external_zip_url,
            self.dataset_directory,
            c_size=512,
            file_type=None,
            remote_fname_name=None,
            force_download=force_download
        )
        if not zip_name:
            raise RuntimeError(f'Error downloading zip file: {zip_name}')

        extraction_path = unzip_file(zip_name, self.dataset_directory, force_unzip=force_unzip, remove_zip=False)
        if not extraction_path:
            raise RuntimeError(f'Error unzipping file file: {zip_name}')

        time_start = time.time()
        physio_files = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        time_end = time.time()
        print(f'Time to find edf files: {time_end - time_start:.4f} seconds')
        print(f'Found {len(physio_files)} EDF files')

        # limiter for debug purposes
        if debug:
            physio_files = physio_files[:8]

        trial_info_list = []
        trial_data_list = []
        num_cpus = mp.cpu_count()
        if use_mp:
            print(f'Using max {num_cpus} processes to reformat {len(physio_files)} files')
            mp_pool = mp.Pool(processes=num_cpus)
            pbar = tqdm(total=len(physio_files), desc=f'Building trial dataframes', file=sys.stdout)
            trial_info_list_lock = threading.Lock()
            trial_data_list_lock = threading.Lock()

            def process_success(future_result):
                pbar.update(1)
                with trial_info_list_lock:
                    trial_info_list.append(future_result[0])
                with trial_data_list_lock:
                    trial_data_list.extend(future_result[1])
                return

            for each_file in physio_files:
                mp_pool.apply_async(
                    self.reformat_as_dataframes, (each_file,),
                    callback=process_success,
                )
            mp_pool.close()
            mp_pool.join()
            pbar.close()
        else:
            print(f'Using a single process to reformat {len(physio_files)} files')
            pbar = tqdm(total=len(physio_files), desc=f'Building trial dataframes', file=sys.stdout)
            for each_file in physio_files:
                each_trial_info, each_trial_samples = self.reformat_as_dataframes(each_file)
                trial_info_list.append(each_trial_info)
                trial_data_list.extend(each_trial_samples)
                pbar.update(1)
            pbar.close()
        self.trial_info_df = pd.DataFrame(trial_info_list)
        self.trial_data_df = pd.DataFrame(trial_data_list)
        return

    def reformat_as_dataframes(self, file_name):
        mne.set_log_level('CRITICAL')  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
        f_basename, _ = os.path.splitext(os.path.basename(file_name))
        subject_name = f_basename[:4]
        entry_trial = f_basename[4:]

        trial_info = {
            'source': self.name,
            'subject': subject_name,
            'trial_type': self.__trial_type_from_name(entry_trial),
            'trial_name': entry_trial,
        }
        trial_id = build_entry_id(trial_info)
        trial_info['id'] = trial_id

        raw_edf = read_raw_edf(file_name, preload=True, verbose='CRITICAL')
        cleaned_edf = self.__clean_raw_edf(raw_edf)
        data = cleaned_edf.get_data()
        np_data = np.transpose(data)
        events_timings, event_indices = mne.events_from_annotations(cleaned_edf)

        sample_list = []
        for sample_idx, each_sample in enumerate(np_data):
            current_event = self.__last_event_before_idx(events_timings, sample_idx)
            sample_entry = {
                'id': trial_id,
                'idx': sample_idx,
                'timestamp': idx_to_time(sample_idx, self.sample_freq),
                'label': current_event
            }
            for dp_idx, dp in enumerate(each_sample.tolist()):
                sample_entry[self.coi[dp_idx]] = dp
            sample_list.append(sample_entry)

        return trial_info, sample_list

    def __last_event_before_idx(self, event_list, sample_idx):
        trimmed_list = []
        for event in event_list:
            if event[0] <= sample_idx:
                trimmed_list.append(event)
        return self.event_names[trimmed_list[-1][2] - 1]

    def __validate_dataset(self) -> bool:
        if self._validated:
            print('Dataset has already been validated')
            return True

        print('Validating dataset')
        if not os.path.isfile(self.trial_info_file):
            print(f'Unable to locate meta file describing this datasource: {self.trial_info_file}')
            return False

        if not os.path.isfile(self.trial_data_file):
            print(f'Unable to locate data file containing the trial samples: {self.trial_data_file}')
            return False

        self._validated = True
        return self._validated


def main():
    """

    :return:
    """
    display_generators = True

    physio_ds = PhysioDataSource(save_method='h5')
    print(physio_ds.subject_names)

    if display_generators:
        sample_list = list(physio_ds)
        print(f'Number of samples: {len(sample_list)}')

        for sample in physio_ds:
            print(f'{sample["idx"]}:{sample["timestamp"]}:'f'{sample["label"]}')

        for sample in physio_ds.downsample_generator(2):
            print(f'{sample["idx"]}:{sample["timestamp"]}:'f'{sample["label"]}')

        for window in physio_ds.window_generator(0.1, 0.1):
            window_head = window.iloc[0]
            window_tail = window.iloc[-1]
            print(f'{window_head["timestamp"]}:{window_tail["timestamp"]}')
    return


if __name__ == '__main__':
    main()
