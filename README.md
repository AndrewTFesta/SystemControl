System Control
=====

This code is in support of the research performed for the thesis requirement for the Master's degree in Computer Science from Rochester Institute of Technology. The various reports can be found in the `/docs/reports` folder, linked directly below.

- [Pre-proposal](docs/reports/Thesis_Pre_Proposal__Data_Representation_for_Motor_Imagery_Classification.pdf)
- [Proposal](docs/reports/Thesis_Proposal__Data_Representation_for_Motor_Imagery_Classification.pdf)
- [Thesis Report](docs/reports/Thesis__Data_Representation_for_Motor_Imagery_Classification.pdf)
- [Signature Page](docs/reports/thesis_signature_page.pdf)

The research was aimed at exploring the various ways in which the data representation affected the ability to perform motor-imagery classification, an inverse problem in neuroscience which attempts to determine a user's intent to move with measured brainwaves.

# License

See the [LICENSE file](LICENSE) for license rights and limitations (MIT).

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

Recent advances in the hardware required for small-scale and non-intrusive methods of measuring brain activity offer an unprecedented level of potential for the development of brain-controlled interfaces (BCI). Where this type of technology used to be accessible only to the professional medical community \cite{bib:bci_challenges, bib:masters:eeg_bci}, non-professionals are now able to approach this domain as a viable method of control. \href{http://neurosky.com/blog/}{NeuroSky} and \href{https://www.emotiv.com/category/news/}{Emotiv} both provide cost-effective boards for recording electroencephalograms (EEG) for developers to use for experiments along with a thriving community for novices and experts alike. \href{https://openbci.com/community/}{OpenBCI} takes this a step further by open-sourcing both the software and the hardware for their boards, the \href{https://shop.openbci.com/collections/frontpage/products/pre-order-ganglion-board}{Ganglion} and the \href{https://shop.openbci.com/collections/frontpage/products/cyton-biosensing-board-8-channel}{Cyton}.

Despite the explosive growth of the field since the early 90's \cite{bib:ultimate_guide_eeg, bib:history_eeg}, drawing meaning from the understanding of the brain remains a difficult challenge of two parts. The first is the challenge of the forward problem which is an attempt to discern the expected outputs from the brain given an initial set of environmental stimuli. The other is the inverse problem which attempts to map the brainwaves back to the stimuli which gave rise to those patterns. The open-source community tends to focus on the inverse problem in an effort to build BCI and other control systems. The forward problem generally left to large research labs with extensive resources. But to create a truly effective BCI, both the forward and inverse problems must be addressed in tandem to ensure that the system operates based on the theoretical truths of the functionality of the brain.

# Usage

# Installation

## Required

- vtk
- pydot
- graphviz
- ffmpeg
- libvips
    https://pypi.org/project/pyvips/

## Optional

- tensorflow-gpu

# Examples
