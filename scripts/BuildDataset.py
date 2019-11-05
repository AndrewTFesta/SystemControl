"""
@title
@description
"""
import os
import shutil
import sys
import time

from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource
from SystemControl.DataTransformer import DataTransformer, CMAP, Interpolation
from SystemControl.utilities import find_files_by_type


def main():

    debug = False

    row_spacing: int = 100
    interp: Interpolation = Interpolation.LINEAR

    start_pad_points = 1
    start_pad_step = 0.1

    duration_points = 5
    duration_step = 0.1

    data_source_list = [PhysioDataSource()]
    start_padding_list: list = [(idx + 1) * start_pad_step for idx in range(0, start_pad_points)]
    duration_list = [(idx + 1) * duration_step for idx in range(0, duration_points)]
    enum_members = list(Interpolation.__members__.values())
    valid_subject_names = data_source_list[0].subject_names

    # todo data only relies on data_source and subject
    # todo slicing only relies on spad and duration
    # todo image creation only relies on interpolation

    num_calls = len(data_source_list)
    num_calls *= len(start_padding_list)
    num_calls *= len(duration_list)
    num_calls *= len(enum_members)
    num_calls *= len(valid_subject_names)

    timings_list = []
    pbar = tqdm(total=num_calls, desc=f'', file=sys.stdout)
    for each_data_source in data_source_list:
        data_transformer = DataTransformer(
            each_data_source, subject=valid_subject_names[0], spacing=row_spacing, cmap=CMAP.rocket_r,
            interpolation=interp,
            start_padding=start_padding_list[0], duration=duration_list[0], debug=debug
        )

        for each_start in start_padding_list:
            data_transformer.set_start_padding(each_start)
            for each_duration in duration_list:
                data_transformer.set_duration(each_duration)
                for each_enum in enum_members:
                    data_transformer.set_interpolation(each_enum)
                    for each_subject in valid_subject_names:
                        try:
                            data_transformer.set_subject(each_subject)
                            base_desc = f'{each_data_source.name}: {each_start}: {each_duration}: ' \
                                        f'{each_enum}: {each_subject}'

                            start_time = time.time()
                            pbar.set_description(f'{base_desc}: Validating existing images')
                            img_paths = find_files_by_type('png', root_dir=data_transformer.base_dir)
                            num_exist_imgs = len(img_paths)

                            pbar.set_description(f'{base_desc}: Slicing data')
                            data_transformer.slice_data()
                            if num_exist_imgs > len(data_transformer.data_slices):
                                shutil.rmtree(data_transformer.base_dir)
                                num_exist_imgs = 0

                            if num_exist_imgs < len(data_transformer.data_slices):
                                pbar.set_description(f'{base_desc}: Building images')
                                data_transformer.build_all_images()
                                pbar.set_description(f'{base_desc}: Saving images')
                                data_transformer.save_images()

                            end_time = time.time()
                            d_time = end_time - start_time
                            timings_list.append(d_time)
                        except Exception as e:
                            print(str(e))
                        finally:
                            pbar.update(1)
    pbar.close()

    out_dir = os.path.join(DATA_DIR, 'output', 'timings')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    timing_fname = os.path.join(out_dir, f'dataset_generation_{num_calls}.txt')
    with open(timing_fname, 'w+') as timing_file:
        for each_timing in timings_list:
            timing_file.write(f'{each_timing}\n')
    return


if __name__ == '__main__':
    main()

########################################
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
#
