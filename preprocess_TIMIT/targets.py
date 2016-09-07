__author__ = 'joerg'

import numpy as np


def get_timit_dict(dic_location):
    #read file with all phonemes (silences are all in line 61)
    file_obj = open(dic_location, 'r')
    phonem_assigment = file_obj.readlines()
    file_obj.close()

    phonemlist_length = phonem_assigment.__len__()
    max_phonem_length = 61 -1 #-1 for array
    #create key value dictionary
    dict_timit = {}
    for i in xrange(phonemlist_length):
        symbol = phonem_assigment[i].split(" ")
        symbol = symbol[0].split("\n")[0]
        dict_timit[symbol] = min(i, max_phonem_length)
    return dict_timit


def get_target(phn_location,dict_timit, nframes, input_size):


    phn_file = open(phn_location, 'r')
    phn_position = phn_file.readlines()
    phn_file.close()

    phn_position_length = phn_position.__len__() -1

    #input_size = int(nframes / float(sample_rate) / float(winstep))
    window_factor = float(nframes) / input_size

    target = np.empty([input_size,61])
    #target_raw = []

    #get first phonem
    phn_count = 0
    low_bound, hight_bound, symbol = phn_position[phn_count].split(" ")
    low_bound = int(low_bound)
    hight_bound = int(hight_bound)
    symbol = symbol.rstrip()

    #go step by step through target vector and add phonem vector
    for i in xrange(input_size):
        threshold =  i * window_factor
        if hight_bound > threshold:
            tarray = np.zeros(61)
            tarray[dict_timit[symbol]] = 1
            target[i] = tarray
            #target_raw.append(symbol+" "+str(int(threshold)))
        else:
            #get next phonem
            phn_count = min(phn_count+1, phn_position_length)
            low_bound, hight_bound, symbol = phn_position[phn_count].split(" ")
            low_bound = int(low_bound)
            hight_bound = int(hight_bound)
            symbol = symbol.rstrip()

            tarray = np.zeros(61)
            tarray[dict_timit[symbol]] = 1
            target[i] = tarray
            #target[i] = dict_timit[symbol]
            #target_raw.append(symbol+" "+str(int(threshold)))

    return target

