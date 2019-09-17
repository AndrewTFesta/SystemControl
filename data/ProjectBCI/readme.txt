https://sites.google.com/site/projectbci/

If you publish your research making any use of this data, please send us an email so that the usage reference can be added to this page. Thanks
The following EEG datasets were used in this research. All downloads are in Matlab MAT format

Dataset 1 - 1D motion
Info: This subject is a 21 year old, right handed male with no known medical conditions. The EEG consists of actual random movements of left and right hand recorded with eyes closed. Each row represents one electrode. The order of the electrodes is FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ. The recording was done at 500Hz using Neurofax EEG System which uses a daisy chain montage. The data was exported with a common reference using Eemagine EEG. AC Lines in this country work at 50 Hz. This info is also included in the MAT file.

Dataset 2 - 2D motion
Info: This subject is a 21 year old, right handed male with no known medical conditions. The EEG consists of actual random movements of left and right hand recorded with eyes closed. Each row represents one electrode. The order of the electrodes is FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ. The recording was done at 500Hz using Neurofax EEG System which uses a daisy chain montage. The data was exported with a common reference using Eemagine EEG. AC Lines in this country work at 50 Hz.

This data consists of the following movements

1. Three trials left hand forward movement

2. Three trials left hand backward movement

3. Three trials left hand forward movement

4. Three trials left hand forward movement

5. 1 trial imagined left hand forward movement

6. 1 trial imagined left hand backward movement

7. 1 trial imagined right hand forward movement

8. 1 trial imagined right hand backward movement

9. 1 trial left leg movement 10. 1 trial right leg movement

Erratum: In readme variable, please replace 6\50 Hz with 50 Hz.

Additional Info
Following are answers to some of the questions asked by the community for this dataset

1. Different trials are stored in different matlab variables. The variable names specify the type of movement and trial number if there are more than one.
2. An audio queue (spoken word "go") was used to initiate the movement. The subject already knew what movement to perform on next "go".
3. The trials were of slightly different lengths, and they have been trimmed to the length of the shortest trial. last samples from longer trials were dropped.
4. The time length of the trials can be calculated using the sampling rate, 500 samples per second.

5.For the 1-D motion dataset, the subject performs continuous random movement of the left and right hand.

6. The subject did not comtrol their breathing or swallowing.

7. The room was not EM shielded
8. 2-D imagined movements are one continuous movement.