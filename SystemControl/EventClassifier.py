"""
@title
@description
"""

import numpy as np
from mne import events_from_annotations

from SystemControl import DataTransformer, DATABASE_URL
from SystemControl.DataSource import DataSource, SqlDb
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource


class EventClassifier:

    def __init__(self, data_source: DataSource, data_transformer: DataTransformer, subject: int = 1):
        self.data_source = data_source
        self.data_transformer = data_transformer

        self.subject = subject

        self.raw_data = self.data_source.get_mi_right_left(self.subject)
        self.data = self.data_source.get_data(self.raw_data)

        self.annotations = self.data_source.get_annotations(self.raw_data)
        self.events = events_from_annotations(self.raw_data)
        return

    @staticmethod
    def event_from_id(event_dict, event_id):
        for evt_key, evt_val in event_dict.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if evt_val == event_id:
                evt_str = evt_key
                break
        else:
            evt_str = None
        return evt_str

    def build_classification_dataset(self):
        # todo  create images from data slices
        start_padding = 0.1
        end_padding = 0.1
        event_duration = 0.5
        freq = self.raw_data.info['sfreq']

        num_start_samples_padding = int(freq * start_padding)
        num_end_samples_padding = int(freq * end_padding)
        num_samples_per_event = int(freq * event_duration)

        event_list = self.events[0]
        event_types = self.events[1]
        d_transpose = np.transpose(self.data)
        data_slice_list = []
        for event_idx, each_event in enumerate(event_list):
            sample_idx = each_event[0]
            event_id = each_event[2]
            event_str = EventClassifier.event_from_id(event_types, event_id)
            end_sample_idx = sample_idx + num_samples_per_event

            start_slice_idx = sample_idx - num_start_samples_padding
            end_slice_idx = end_sample_idx + num_end_samples_padding
            # only add to slice_list if data range is valid (non negative and not beyond bounds of d_transpose
            if start_slice_idx >= 0 and end_slice_idx < len(d_transpose):
                data_slice = d_transpose[start_slice_idx:end_slice_idx]
                data_slice_list.append(data_slice)
                print(f'{start_slice_idx}:{end_slice_idx}, Event: {event_str}, Data points: {data_slice.shape}')
            else:
                print(f'Data slice falls outside valid bounds ({0}:{len(d_transpose)}): '
                      f'({start_slice_idx}:{end_slice_idx})')
        return data_slice_list

    def train(self):
        # todo
        return

    def predict(self):
        # todo
        return

    def next_event_vals(self):
        event_onset = self.events[0]
        print(event_onset)
        return


def main():
    subject = 1

    db_path = DATABASE_URL
    database = SqlDb.SqlDb(db_path)
    physio_data_source = PhysioDataSource(database)
    data_transformer = DataTransformer.DataTransformer()

    event_classifier = EventClassifier(physio_data_source, data_transformer, subject)
    event_classifier.build_classification_dataset()
    return


if __name__ == '__main__':
    main()

########################################

# # Read epochs (train will be done only between 1 and 2s)
# # Testing will be done with a running classifier
# epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
# epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
# labels = epochs.events[:, -1] - 2
#
# ########################################
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
