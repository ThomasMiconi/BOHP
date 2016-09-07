# Backpropagation of Hebbian Plasticity

This is the code for backpropagation of Hebbian plasticity, as described in an upcoming paper.

This code is built upon Andrej Karpathy's [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086). Any bugs and inefficiencies are entirely my own.

The code is entirely contained in the bohp.py scritp. Other scripts are for cluster submission or making figures. This code requires only Python 2 and numpy.

You can run the script as-is, or modify some parameters with command-line arguments. Default parameters are the ones used in the paper for the 'uncorrelated' problem. 

Examples (from IPython):

    %run bohp.py
    %run bohp.py LEARNPERIOD 0 YSIZE 3
    %run bohp.py ETA 0.003 MAXDW 0.01 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95

The only difference with the paper is that we now use a logistic (strictly-positive) output for the final layer. 




