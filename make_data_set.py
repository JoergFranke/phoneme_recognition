#!/usr/bin/env python
""" Makes a data set in klepto files with MEL frequencies from the TIMIT speech corpus"""

from __future__ import print_function

import os

import klepto

from preprocess_TIMIT.targets import get_target, get_timit_dict
from preprocess_TIMIT.features import get_features
from preprocess_TIMIT.speakers import get_speaker_lists


# Location of the TIMIT speech corpus
#########################################################
###             Add path to TIMIT corpus              ###
#########################################################
#rootDir = "/path/to/TIMIT/"
rootDir = "/media/joerg/hddred/development/datasets/TIMIT/"


# Location of the target data set folder
drainDir = "data_set/"


####################################
# pre process parameters

# Mel-Frequency Cepstrum Coefficients, default 12
numcep=12 #40
# the number of filters in the filterbank, default 26.
numfilt =26 #40

# the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
winlen = 0.025
# the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
winstep = 0.01
# use  one or second order derivation
grad = 1

para_name = "mfcc" + str(numcep) + "-" + str(numfilt) + "win" + str(int(winlen*1000)) + "-" +str(int(winstep*1000))

train_speaker, valid_speaker = get_speaker_lists(rootDir + "TRAIN/")
print(train_speaker)
print(train_speaker.__len__())
print(valid_speaker)
print(valid_speaker.__len__())
dic_location = "preprocess_TIMIT/phonemlist"

timit_dict = get_timit_dict(dic_location)

train_set_x = []
train_set_y = []
valid_set_x = []
valid_set_y = []

dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']
for d in dirlist:
    for dirName, subdirList, fileList in os.walk(rootDir + "TRAIN/" + d + "/"):
        print('Found directory: %s' % dirName)

        path,folder_name = os.path.split(dirName)
        print('Speaker: ' + folder_name)
        if folder_name.__len__() >= 1:
            temp_name = ""
            for fname in sorted(fileList):
                name = fname.split(".")[0]
                if name != temp_name:
                    temp_name = name
                    print('\t%s' % dirName+"/"+name)
                    wav_location = dirName+"/"+name+".WAV"
                    phn_location = dirName+"/"+name+".PHN"
                    feat, frames, samplerate = get_features(wav_location, numcep, numfilt, winlen, winstep, grad)
                    print(feat.shape)
                    input_size = feat.shape[0]
                    target = get_target(phn_location,timit_dict, frames, input_size)
                    if folder_name in train_speaker:
                        train_set_x.append(feat)
                        train_set_y.append(target)
                    elif folder_name in valid_speaker:
                        valid_set_x.append(feat)
                        valid_set_y.append(target)
                    else:
                        assert False, "unknown name"



print("write valid set")
print("valid set length: " + str(valid_set_x.__len__()))
file_name = drainDir + "timit_" + "valid_" + "xy_" + para_name + ".klepto"
print("valid set name: " + file_name)
d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
d['x'] = valid_set_x
d['y'] = valid_set_y
d.dump()
d.clear()

print("write train set")
print("train set length: " + str(train_set_x.__len__()))
file_name = drainDir + "timit_" + "train_" + "xy_" + para_name + ".klepto"
print("train set name: " + file_name)
d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
d['x'] = train_set_x
d['y'] = train_set_y
d.dump()
d.clear()


test_set_x = []
test_set_y = []

for d in dirlist:
    for dirName, subdirList, fileList in os.walk(rootDir + "TEST/" + d + "/"):
        print('Found directory: %s' % dirName)

        path,folder_name = os.path.split(dirName)
        print('Speaker: ' + folder_name)
        if folder_name.__len__() >= 1:
            temp_name = ""
            for fname in sorted(fileList):
                name = fname.split(".")[0]
                if name != temp_name:
                    temp_name = name
                    print('\t%s' % dirName+"/"+name)
                    wav_location = dirName+"/"+name+".WAV"
                    phn_location = dirName+"/"+name+".PHN"
                    feat, frames, samplerate = get_features(wav_location, numcep, numfilt, winlen, winstep, grad)
                    print(feat.shape)
                    input_size = feat.shape[0]
                    target = get_target(phn_location,timit_dict, frames, input_size)
                    test_set_x.append(feat)
                    test_set_y.append(target)



print("write test set")
print("test set length: " + str(test_set_x.__len__()))
file_name = drainDir + "timit_" + "test_" + "xy_" + para_name + ".klepto"
print("test set name: " + file_name)
d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
d['x'] = test_set_x
d['y'] = test_set_y
d.dump()
d.clear()