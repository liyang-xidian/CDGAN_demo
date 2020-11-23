# CDGAN_demo
Speech Separation

CDGAN based speech separation demo (developed by Dr. liyang)

Note: Matlab v2018 or later should be installed in your PC first. 
This zip contains the CDGAN model trained by us.
The speech are selected from the TIMIT database. We selected 10 speech from the TIMIT database as examples, you can choose any number of other speech and put them into the speech folder(Any file name.Wav is OK).

separation steps:
1 Run CDGAN_demo.m and the program will randomly select a speech from all of the speech you put in the speech folder and mix it with a random noise in the noise92 database.
2 The program will first give the time domain waveform and spectrum of target speech, interference, convolution mixed speech and additive mixed speech respectively. Then the program will give the time domain waveform and spectrum of reverberation suppression and speech enhancement results.
