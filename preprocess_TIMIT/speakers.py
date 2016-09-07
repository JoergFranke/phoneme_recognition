__author__ = 'joerg'

import os
import numpy as np


def get_speaker_lists(rootDir):

    np.random.seed(100)

    dirlist = ['DR5', 'DR6', 'DR7', 'DR3', 'DR2', 'DR1', 'DR4', 'DR8']

    train_speaker = []
    valid_speaker = []

    for i in dirlist:

        region_speakers = []

        for dirName, subdirList, fileList in os.walk(rootDir + i + "/"):
            #print(dirName)
            path,folder_name = os.path.split(dirName)
            if folder_name.__len__() >= 1:
                region_speakers.append(folder_name)

        len = region_speakers.__len__()
        valid_len = int(round(len * 0.1))
        random_valid = np.random.random_integers(0,region_speakers.__len__()-1,valid_len)
        random_train = np.delete(np.arange(0,region_speakers.__len__()),random_valid)
        region_speakers = np.asarray(region_speakers)

        train_speaker = train_speaker + list(region_speakers[random_train])
        valid_speaker = valid_speaker + list(region_speakers[random_valid])

    return train_speaker, valid_speaker
