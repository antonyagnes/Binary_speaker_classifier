# Binary_speaker_classifierDefining the problem
	Audio analysis is one of the growing areas since the development of Deep Learning.
The problem is to build a binary classifier to identify two different speakers.

Objectives

	Use the .wav files to identify the speaker. The audio files for two different sapeakers are taken into consideration. We say 1 for Jackson and 0 for Nicolas.

Install 

Librosa
Glob
Radom
Numpy
Keras
Pandas

Run

	In the terminal or window navigate to the top level of the project directory and run the following command
python speaker_prediction.py

Data

A simple audio/speech dataset consisting of recordings of spoken digits in wav files at 8kHz. The recordings are trimmed so that they have near minimal silence at the beginnings and ends.
FSDD is an open dataset, which means it will grow over time as data is contributed. Thus in order to enable reproducibility and accurate citation in scientific journals the dataset is versioned using git tags.
Current status
•	3 speakers
•	1,500 recordings (50 of each digit per speaker)
•	English pronunciations

Target variable

1 for jackson
0 for nicolas
