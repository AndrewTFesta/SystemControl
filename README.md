System Control
=====

This code is in support of the research performed for the thesis requirement for the Master's degree in Computer Science from Rochester Institute of Technology. The various reports can be found in the `/docs/reports` folder, linked directly below.

- [Pre-proposal](docs/reports/Thesis_Pre_Proposal__Data_Representation_for_Motor_Imagery_Classification.pdf)
- [Proposal](docs/reports/Thesis_Proposal__Data_Representation_for_Motor_Imagery_Classification.pdf)
- [Thesis Report](docs/reports/Thesis__Data_Representation_for_Motor_Imagery_Classification.pdf)
- [Defense Presentation](docs/reports/Defense.pdf)
- [Signature Page](docs/reports/thesis_signature_page.pdf)

The research was aimed at exploring the various ways in which the data representation affected the ability to perform motor-imagery classification, an inverse problem in neuroscience which attempts to determine a user's intent to move with measured brainwaves.

# License

See the [LICENSE file](LICENSE) for license rights and limitations (MIT).

# Index

- [Abstract](#abstract)
- [Dedication](#dedication)
- [Acknowledgements](#acknowledgements)
- [Introduction](#introduction)
- [Datasets](#datasets)
    - [PhysioNet](#physionet)
    - [Manually Recorded](#manually-recorded)
- [Next Steps](#next-steps)
    - [Hardware Development](#hardware-development)
        - [Custom EEG Board](#custom-eeg-board)
        - [Active Electrodes](#active-electrodes)
    - [Algorithmic Development](#algorithmic-development)
        - [Data Cleaning and Noise Isolation](#data-cleaning-and-noise-isolation)
        - [Field Reconstruction](#field-reconstruction)
- [Code and Dataset Repositories](#code-and-dataset-repositories)
    - [OpenBCI](#openbci)
    - [BCI Competition IV](#bci-competition-iv)
    - [EEG Datasets](#eeg-datasets)
- [Neuroscience Learning Resources](#neuroscience-learning-resources) 
    - [The typical M/EEG workflow](#the-typical-meeg-workflow)
    - [EEG: The Ultimate Guide](#eeg-the-ultimate-guide)
    - [What is EEG (Electroencephalography) and How Does it Work?](#what-is-eeg-electroencephalography-and-how-does-it-work)
    - [Reading Minds with Deep Learning](#reading-minds-with-deep-learning)
    - [Building a mind-controlled drone](#building-a-mind-controlled-drone)

# Abstract

While much progress has been made towards the advancement of brain-controlled interfaces (BCI), there remains an information gap between the various domains involved in progressing this area of research. Thus, this research seeks to address this gap through creation of a method of representing brainwave signals in a manner that is intuitive and easy to interpret for both neuroscientists and computer scientists. This method of data representation was evaluated on the ability of the model to accurately classify motor imagery events in a timely manner.

The proposed data representation of electroencephalographic signals in the form of signal images was found to be able to perform adequately in the task of motor-imagery. However, the amount of time to record enough samples was on the scale of a fifth of a second following the onset of an input from the user. This time delay represents the minimum window size needed to classify the event, meaning that to reduce this delay would require a fundamental shift in the data that is acted upon to perform classification or to generate the signal images. Furthermore, the system performed better than expected, even in the face of random data, suggesting that the system may be relying on some external factor or undesired artifact present in the data in order to perform its task.

The strength of this approach came from its ability to be understood, visually examined, and altered in near-real-time in order to explore the data captured during a recording session. This was done after data had been recorded and involved altering sets of configuration parameters that affect the computations that go into generating a signal image. Namely, this included the window size, the function used to interpolate between two adjacent data points, and the amount of overlap of the windows. Effectively, this allows a researcher to playback the signal in an intuitive manner, watching for large shifts or edges in the images in order to detect large changes in the underlying data stream. Thus, while this approach may be unsuited for the task of classification, it would be an effective tool for conducting exploratory data analysis.

# Dedication

> *To all the researchers out there who just want to have fun and explore the possibilities. And if something good comes of it, then all the better.*

# Acknowledgements

As with any extensive effort, this work could not have been possible without the help and support of many others. I could not have done it by myself, and I make no claims about having the aptitude to do so.

To my advisor, Professor Jeremy Brown, who not only served to provide insight and feedback for this research, but also guided my learning over the past several years I have been at RIT for my Bachelor's and Master's degrees. He continually dealt with my overly stubborn nature and yet, against all better judgement, still agreed to act as my advisor for my thesis research.

For the other members of my committee, Professor Ifeoma Nwogu and Professor Philip White. Beyond your technical teachings, you helped me to improve my writing skills -- both in and out of the classroom. And like Professor Brown, you were willing to humor me when I was being obstinate. Or when I was just looking to chat after class or in the hallways. The little details often add up to more than expected.

And finally to my other mentors -- Jackie Corricelli, Craig Paradis, Jeffrey Bierline, and Napoleon Paxton -- and my parents. You helped shape my perspective and knowledge of the world and encouraged me to view failure just as importantly as success. Without you, I would not have made it to where I am today. You have only yourselves to blame for what you have unleashed on the world.

# Introduction

Recent advances in the hardware required for small-scale and non-intrusive methods of measuring brain activity offer an unprecedented level of potential for the development of brain-controlled interfaces (BCI). Where this type of technology used to be accessible only to the professional medical community, non-professionals are now able to approach this domain as a viable method of control. [NeuroSky](http://neurosky.com/blog/) and [Emotiv](https://www.emotiv.com/category/news/) both provide cost-effective boards for recording electroencephalograms (EEG) for developers to use for experiments along with a thriving community for novices and experts alike. [OpenBCI](https://openbci.com/community/) takes this a step further by open-sourcing both the software and the hardware for their boards, the [Ganglion](https://shop.openbci.com/collections/frontpage/products/pre-order-ganglion-board) and the [Cyton](https://shop.openbci.com/collections/frontpage/products/cyton-biosensing-board-8-channel).

Despite the explosive growth of the field since the early 90's, drawing meaning from the understanding of the brain remains a difficult challenge of two parts. The first is the challenge of the forward problem which is an attempt to discern the expected outputs from the brain given an initial set of environmental stimuli. The other is the inverse problem which attempts to map the brainwaves back to the stimuli which gave rise to those patterns. The open-source community tends to focus on the inverse problem in an effort to build BCI and other control systems. The forward problem generally left to large research labs with extensive resources. But to create a truly effective BCI, both the forward and inverse problems must be addressed in tandem to ensure that the system operates based on the theoretical truths of the functionality of the brain.

# Datasets

There are a myriad of challenges to overcome when performing the recording sessions necessary for gathering data for BCI systems. For the non-neuroscientist, there exists a further issue in that it can be difficult to verify if a recording session gathered adequate clean data and how to inspect the signals for abnormalities and artifacts.

## [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/)

In order to reduce uncertainty as to whether errors may be due to the data representation or the recorded data, an external dataset was used to first build out and evaluate the proposed method of data representation. This dataset, called the `EEG Motor Movement/Imagery Dataset`, is provided for research use by PhysioNet. Not only is this dataset expertly collected and meticulously verified, but it is also used by several solutions for evaluation of BCI systems making it a perfect candidate not only for verifying the efficacy of the proposed method for classification of motor imagery events, but also for comparison against current techniques used for performing the same task.

This dataset was built using the BCI2000 system, which is a mid to high level system compared to the Ganglion, the OpenBCI board used for manual data collection. The BCI2000 system has 64 channels, compared to the Ganglion's 4 channels, and is more aimed towards neuroscientist researchers rather than the open-source and non-professional communities. The actual data recorded is comprised of 6 different trial types for 109 different subjects, and each trial is either 2 or 3 minutes in length, depending on the trial type. Each of the baseline trials are 2 minutes in length, while the remaining trials are 3 minutes in length.

## Manually Recorded

Part of the desired outcome for this research was to explore the ability to use the proposed data format for development of a BCI system. As such, there has to be at least some capability to record and analyze EEG activity in real-time. This was done using the Ganglion board from OpenBCI. The user was connected to the board using passive electrodes and presented with a random stimulus on a timer. They were then expected to imagine moving either their right or left hand, based on the stimulus presented. Additionally, there was a "rest" prompt, to which they were expected to just relax and not imagine any action.

As the Physio dataset is the reference dataset, the method for manually collecting data seeks to emulate that process as much as possible. Specifically, there are several key decisions that this protocol copies from the creators of the Physio dataset. There are three prompts that are presented to the subject: 'stop', 'left', and 'right'. The subject was instructed to continue to imagine moving the hand that corresponds to the given prompt in the case of the latter two prompts. In the case of the 'stop' prompt, they were to relax and not imagine moving either hand. Both the EEG samples and the events were recorded, and a single sample (which is comprised of three values -- one from each electrode) is marked as the current event. An important thing to note is that this protocol follows the decision to record when the stimulus was presented rather than when the user reacted to the event.

# Next Steps

While the research performed proves promising for this form of data representation for motor imagery classification, the work herein lays the groundwork for future development. Significant work remains in order to viably use it as a means of input for an effective and real-time BCI system. Part of this works deals not with the software and algorithmic approach, but in improving the hardware and the data acquisition capabilities of the system. In general, the next steps can be broken down into two focus areas: hardware progression and algorithmic development. While both are important as to the development of BCI systems, only one step should be taken at a time in order to ensure a solid base for development. The following suggestions for development first seek to address the issue of data acquisition and then enhance the efficacy of the data representation for more specialized approaches.

## Hardware Development

The hardware boards for performing EEG collection have come a long way in the past decade. However, cost is often directly tied to spatial resolution. For the task of motor-imagery, high spatial resolution is not as important a factor, but it is still a desired characteristic of any BCI or EEG system.

### Custom EEG Board

Creating a custom board for data acquisition is meant to serve two purposes. The first is an attempt to reduce the cost and improve upon the spatial resolution. The second attempts to increase the control offered by creating the system from scratch. While not a trivial task, it is made somewhat easier by the increase of open-source projects in the space. The [circuit schematic](https://github.com/OpenBCI/Ganglion_Hardware_Design_Files/blob/master/Ganglion_SCH.pdf) for the Ganglion is open-sourced by OpenBCI and has an active community aiding in its development. This is possible as the main business model of OpenBCI seems to not be focused with the intellectual property of the boards, but rather the convenience of having them build and provide the board with minimal oversight required of the hobbyist. Unfortunately, active work on the hardware of the board seems to have stalled.

### Active Electrodes

Beyond creating a board that offers at least comparable initial performance to the Ganglion or the Cyton, the electrodes used for data acquisition could theoretically be greatly improved by using active electrodes versus the passive electrodes provided when purchasing a board or headset from OpenBCI. Where passive electrodes essentially act as simple probes measuring the electrical potential and sending this signal to the board for amplification, active electrodes perform a level of amplification at the point of collection. This effectively increases the signal-to-noise ratio of the data as the effect of ambient and environmental noise is less pronounced with respect to the signal after having traveled along the wire of the electrode to the board.

## Algorithmic Development

Where the hardware aspect of development looks to record cleaner and more precise data more directly, algorithmic techniques can be explored which may be able to address issues ranging from data cleaning, exploration, classification, and testing.

### Data Cleaning and Noise Isolation

A point that cannot be emphasized enough is there is no substitute for clean data. The age-old adage is once again proven correct: garbage in means garbage out. This holds particularly true in domains where the data is inherently dirty and difficult to work with. While ideally this data would be collected as cleanly as possible, post-processing is possible which is able to isolate the signal from the noise after it has been recorded. The work of this research took several steps to perform this data cleaning, including filtering out specific frequency ranges; However, the issue of identifying artifacts in the data was not addressed.

Normally, this task would be undertaken by a subject matter expert who would mark bad regions of the data stream for removal. Another approach is to consider that most EEG signals are non-Gaussian in nature, meaning that principal component analysis will likely not be an effective tool to separate the signals. Instead, it is possible to use independent component analysis to separate overlapping events, to remove line noise from the data, and to automatically detect artifacts that may be present due to muscle movement.

### Time-sequence Classification

 When constructing the images, the time-series nature of the problem was handled by concatenating multiple samples together in order to form a 2D image of a signal over time. A better approach may be to use an architecture that is particularly suited for timeseries based classification, such as a recurrent neural network. While it may not prove more effective to use this other type of architecture for classification of these signals due to the increased difficulties in training such a network, it would provide additional insight into the strengths and weaknesses of the data representation, particularly with respect to the effect of the interpolation function and its ability to recreate the electromagnetic fields at play.

### Field Reconstruction

Taking a step back from the end result of a signal image, one of the strengths of the data representation is its ability to interpolate between discrete points in order to partially recreate the electromagnetic field produced by the brain. In the research conducted, only the $C3$, $CZ$, and $C4$ electrode positions were used. This allowed for the interpolation function to operate on the points as if they were on the same x-y coordinate plane due to the fact that these locations are next to each other laterally on the head. However, increasing the spatial resolution of the system breaks this lateral assumption. Remedying the situation only requires having the interpolation function operate on the points in 3D space rather than 2D space. The signal image then becomes more akin to a classic montage or band-plot representation employed by neuroscience domain experts. Feeding it to a classifier then either requires use of a recurrent convolutional neural network or altering the convolutional filter from a 2D filter to a 3D filter, both of which are readily supported by Tensorflow.

# Code and Dataset Repositories

Lists and briefly describes repositories containing code or datasets that may be useful when performing EEG research or attempting to build a brain-computer interface.

## [OpenBCI](https://github.com/OpenBCI)

OpenBCI open-sources all the hardware and software they develop. This GitHub profile contains multiple repositories containing documentation, guides, and code that can be used to aid in building and using their products.

## [BCI Competition IV](http://www.bbci.de/competition/iv/)

The BCI competition was developed to address the issue of high-quality free EEG datasets. It aims to provide a way for researchers and hobyists to validate their signal processing techniques and improve on existing classification techniques. The last dataset was released in 2006 and was featured at NeurIPS (NIPS) 2008.

## [EEG Datasets](https://github.com/meagmohit/EEG-Datasets)

This repository contains links to multiple datasets for many different applications of EEG recordings, including motor-imagery classification.

# Neuroscience Learning Resources

Various links that contain useful information for learning the neuroscience behind brain-controlled interfaces. The content of the earlier links tend to contain mainly theoretical explanations while the later links are more focused towards short guides, tutorials, and projects.

## [The typical M/EEG workflow](https://mne.tools/stable/overview/cookbook.html)

MNE is an open-source Python library for performing M/EEG data analysis. This page, specifically, details the common steps taken and covers the basic considerations that must be taken into account while performing such an analysis. Much of their guide is focused on use of their specific package, but the concepts can be readily applied outside of the framework as well.

## [EEG: The Ultimate Guide](http://neurosky.com/biosensors/eeg-sensor/ultimate-guide-to-eeg/)

Neurosky is another low-cost EEG option that is often used to build out BCI systems. This page details some of the background and basics regarding electroencephalography. In particular, it provides a brief overview of the history of the field and goes into some potential applications in which the technology can be applied.

## [What is EEG (Electroencephalography) and How Does it Work?](https://imotions.com/blog/what-is-eeg/)

IMotions is a purveyor of mid-range EEG headsets as well as resources aimed to help people learn what EEG is and how to use it. This page offers a brief overview of different brainwave patterns and how they can be interpreted from a neuroscience perspective.

## [Reading Minds with Deep Learning](https://blog.floydhub.com/reading-minds-with-deep-learning)

Samuel Lynn-Evans' guide on using convolutional neural networks is fairly thorough, both in terms of explaining the neuroscience behind interpreting brain-wave patterns as well as on how to use the neural network to classify the signals. It uses the [Grasp-and-lift dataset from Kaggle](https://www.kaggle.com/c/grasp-and-lift-eeg-detection/discussion/16479) and attempts to detect which hand is performing the action.

## [Building a mind-controlled drone](https://gear.dev/building-a-mind-controlled-drone)

Jon Gear's implementation of a drone shows how to use two different EEG headsets: the Cyton from OpenBCI and the Mindwave from Neurosky. The blog post walks through setting up a Parrot drone to be controlled by an EEG headset. It is written and described primarily in Javascript, but all the concepts are thoroughly explained. [The GitHub repository]((https://github.com/jongear/mindcraft)) also contains a set of slides and a video demonstrating how to get everything working.
