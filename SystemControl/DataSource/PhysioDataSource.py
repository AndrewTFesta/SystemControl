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
import os
import sys
import time
import urllib.parse

import mne
import numpy as np
from mne.io import read_raw_edf
from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import SubjectEntry, SampleEntry, EventEntry, DataSource
from SystemControl.utilities import download_large_file, unzip_file, find_files_by_type, idx_to_time


def int_to_subject_str(subject_int):
    return f'S{subject_int:03d}'


def int_to_run_str(run_int):
    return f'R{run_int:02d}'


class PhysioDataSource(DataSource):

    def __init__(self, subject: str = None, log_level: str = 'CRITICAL'):
        mne.set_log_level(log_level)  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
        super().__init__(log_level)

        self.name = 'Physio'
        self.sample_freq = 160
        self.subject_names = [int_to_subject_str(each_subject) for each_subject in list(range(1, 110))]
        self.trial_mappings = {
            'baseline_open': [int_to_run_str(each_run) for each_run in [1]],
            'baseline_closed': [int_to_run_str(each_run) for each_run in [2]],
            'motor_execution_right_left': [int_to_run_str(each_run) for each_run in [3, 7, 11]],
            'motor_execution_hands_feet': [int_to_run_str(each_run) for each_run in [5, 9, 13]],
            'motor_imagery_right_left': [int_to_run_str(each_run) for each_run in [4, 8, 12]],
            'motor_imagery_hands_feet': [int_to_run_str(each_run) for each_run in [6, 10, 14]]
        }
        self.trial_types = list(self.trial_mappings.keys())
        self.selected_trial_type = self.trial_types[4]
        self.event_names = ['T0', 'T1', 'T2']
        self.ascended_being = subject if subject else self.subject_names[0]

        self._validated = False
        if not self.__validate_dataset():
            subject_entries = self.__download_dataset()
            self.save_data(subject_entries, human_readable=True)

        self.load_data()
        self.set_subject(self.ascended_being)
        return

    def __clean_raw_edf(self, raw_edf):
        cleaned_raw_edf = raw_edf.copy()
        cleaned_raw_edf.rename_channels(lambda x: x.strip('.'))
        cleaned_raw_edf.pick_channels(self.coi)
        cleaned_raw_edf = cleaned_raw_edf.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        return cleaned_raw_edf

    def __download_dataset(self, force_download=False, force_unzip=False):
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
        print(f'Time to find json files: {time_end - time_start:.4f} seconds')
        print(f'Found {len(physio_files)} JSON files')

        subject_entry_list = []
        for each_file in tqdm(physio_files, desc=f'Reformatting as SubjectEntry objects', file=sys.stdout):
            f_basename, _ = os.path.splitext(os.path.basename(each_file))
            entry_subject = f_basename[:4]
            entry_trial = f_basename[4:]

            raw_edf = read_raw_edf(each_file, preload=True, verbose='CRITICAL')
            cleaned_edf = self.__clean_raw_edf(raw_edf)
            data = cleaned_edf.get_data()
            np_data = np.transpose(data)

            sample_list = []
            for sample_idx, each_sample in enumerate(np_data):
                sample_data = {
                    self.coi[dp_idx]: dp
                    for dp_idx, dp in enumerate(each_sample.tolist())
                }
                sample_entry = SampleEntry(
                    idx=sample_idx,
                    timestamp=idx_to_time(sample_idx, self.sample_freq),
                    data=sample_data
                )
                sample_list.append(sample_entry)

            events_timings, event_indices = mne.events_from_annotations(cleaned_edf)
            event_list = []
            for each_event in events_timings:
                event_entry = EventEntry(
                    idx=int(each_event[0]),
                    timestamp=idx_to_time(each_event[0], self.sample_freq),
                    event_type=self.event_names[each_event[2] - 1]
                )
                event_list.append(event_entry)

            subject_entry = SubjectEntry(
                path=each_file, source_name=self.name, subject=entry_subject,
                trial=entry_trial, samples=sample_list, events=event_list
            )
            subject_entry_list.append(subject_entry)
        return subject_entry_list

    def __validate_dataset(self) -> bool:
        if self._validated:
            print('Dataset has already been validated')
            return True

        missing_files = []
        total_req_files = 0
        time_start = time.time()
        for subject in self.subject_names:
            subject_dir = os.path.join(self.dataset_directory, f'{subject}')
            found_json_files = find_files_by_type(file_type='json', root_dir=subject_dir)
            req_json_fnames = [
                os.path.join(subject_dir, f'{trial_name}.json')
                for trial_name in [int_to_run_str(each_run) for each_run in range(1, 15)]
            ]
            set_found_files = set(found_json_files)
            set_req_files = set(req_json_fnames)
            missing_files.extend(set_req_files - set_found_files)
            total_req_files += len(set_req_files)
        time_end = time.time()
        print(f'Time to validate dataset: {time_end - time_start:.4f} seconds')
        print(f'Missing {len(missing_files)} file(s) of {total_req_files} required files')

        self._validated = len(missing_files) == 0
        return self._validated


def main():
    """

    :return:
    """
    physio_ds = PhysioDataSource()
    print(physio_ds.subject_names)

    for sample, event in physio_ds:
        print(f'{sample["idx"]}:{sample["timestamp"]:<10.4f}::'
              f'{event["idx"]:>7d}:{event["timestamp"]}:{event["event_type"]}')
    return


if __name__ == '__main__':
    main()
