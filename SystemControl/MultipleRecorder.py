"""
@title
@description
"""
import multiprocessing as mp

from SystemControl import TrialRecorder


def main():
    rec_type = 'disconnected'
    mult_rec_args = {
        'record_length': 120,
        'subject_name': None,
        'session_type': 'motor_imagery_right_left',
        'stimulus_delay': 5,
        'jitter': 0.2
    }

    if rec_type is 'disconnected':
        ################################################
        trial_count_range = list(range(1, 11, 2))
        ################################################
        for trial_count in trial_count_range:
            subject_name = f'disconnected_{trial_count:02d}'
            mult_rec_args['subject_name'] = subject_name
            for trial_num in range(0, trial_count):
                rec_proc = mp.Process(target=TrialRecorder.main, args=(mult_rec_args,))
                rec_proc.start()
                rec_proc.join()
    elif rec_type is 'random':
        ################################################
        trial_count_range = range(1, 6)
        ################################################
        for trial_count in trial_count_range:
            mult_rec_args['subject_name'] = 'random'
            rec_proc = mp.Process(target=TrialRecorder.main, args=(mult_rec_args,))
            rec_proc.start()
            rec_proc.join()
    return


if __name__ == '__main__':
    main()
