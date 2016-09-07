# Phoneme Recognition with TIMIT Speech Corpus


<!--- [![Build Status](https://travis-ci.org/joergfranke/rnnfwk.svg?branch=master)](https://travis-ci.org/joergfranke/rnnfwk) --->
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/joergfranke/phoneme_recognition/blob/master/LICENSE.txt)
[![Python](https://img.shields.io/badge/python-2.7-yellow.svg)](https://www.python.org/download/releases/2.7/)
[![Theano](https://img.shields.io/badge/theano-0.8.2-yellow.svg)](http://deeplearning.net/software/theano/)

## About
This repository contains a recurrent neural network based phoneme recognition. 


## Requirements

- The preprocess needs the SND-file library. For example you can install it from the apt-get repository. 
```bash
sudo apt-get install libsndfile-dev
```
- This phoneme recognition model uses the rnnfwk RNN-Framework which needs to be installed. 
```bash
git clone https://github.com/joergfranke/rnnfwk.git
cd rnnfwk
python setup.py install
```
- Furthermore the model requires the packages listed in the `requirements.txt`. 



## How to use

### Step 1: Make data set

This step contains the whole preprocess and creates a data set in the form of a list with sequenzes of features and targets.


data format 
list of sequences

### Step 2: Train recurrent neural network model



### Step 3: Evaluate exercised model



## 

## Acknoledgement


