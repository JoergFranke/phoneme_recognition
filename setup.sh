#!/bin/bash

# Proposal for setup phoneme recognition

if [ "phoneme_recognition" ==  "${PWD##*/}" ]
then
    # Install SND file format library
    sudo apt-get install libsndfile-dev

    # Setup vistual environment
    virtualenv venv
    source venv/bon/activate
    pip install --upgrade pip
    pip install numpy
    pip install -r requirements
    deactivate

    # Install RecNet
    git clone https://github.com/joergfranke/recnet.git
    cd recnet
    ../venv/bin/python setup.py install
else
    echo "Please go in phoneme_recognition directory"
fi



