# DJ-Set-Bot
An unsupervised music information retrieval system for dj set building and track recommendation. 
https://www.mixesdb.com/
http://www.tuneid.com/
https://www.be-at.tv/

# research papers
- https://hal.science/hal-02172427/document

# Feature Generation from Essentia model

## Environment setup 
To properly handle large files like the model weights and other things in the future you must install git lfs, follow the instructions below: 
https://git-lfs.com/

## Using feature models from Essentia
- install wsl if using windows
- create python virtual environment
- activate env and `pip3 install essentia-tensorflow`
- unzip essentiaModel.zip 
- run test.py and input a path to an mp3
    - it expects a standard 320kps quality one, but you can change sample rate in test.py 
- ignore cuda warnings and if it works it will paste a bunch of numbers
