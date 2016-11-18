#!/bin/bash

# Proposal for setup phoneme recognition

if [ "phoneme_recognition" ==  "${PWD##*/}" ]
then
    # Install SND file format library
    sudo apt-get install libsndfile-dev

    # Setup vistual environment
    virtualenv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install --upgrade setuptools
    pip install numpy
    pip install -r requirements.txt
    deactivate

    # Install RecNet
    git clone https://github.com/joergfranke/recnet.git
    cd recnet
    ../venv/bin/python setup.py install
else
    echo "Please go in phoneme_recognition directory"
fi


