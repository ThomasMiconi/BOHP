# Backpropagation of Hebbian Plasticity

This is the commented code for backpropagation of Hebbian plasticity, as described in the Arxiv preprint [Backpropagation of Hebbian plasticity for lifelong learning](https://arxiv.org/abs/1609.02228).

This code simulates a simple fear conditioning task: the network must learn
to determine which of two stimuli is associated with a "pain" signal, and to
produce output 1 whenever that stimulus is present (even in the absence of
the pain signal), and 0 otherwise. Which of the two possible stimuli is associated with the
pain signal changes unpredictably from one episode to the next, but remains
stable within each episode. The BOHP algorithm optimizes both the weight and the
plasticity of all connections, so that during each episode, the network
 quickly learns the necessary associations. See the preprint for details.

This code is built upon Andrej Karpathy's [min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086). Any bugs and inefficiencies are entirely my own.

The code is entirely contained in the `bohp.py` script. Other scripts are for cluster submission or making figures. This code requires only Python 2 and numpy.

You can run the script as-is, or modify some parameters with command-line arguments. Default parameters are the ones used in the paper for the 'uncorrelated' problem. 
The program will store the mean absolute errors of each episode in errs.txt, as well as other data (weights, etc.) in data.pkl. See code for details.

Examples (from IPython):

    %run bohp.py
    %run bohp.py LEARNPERIOD 0 YSIZE 3
    %run bohp.py ETA 0.003 MAXDW 0.01 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95

The only difference with the paper is that we now use a logistic (strictly-positive) output for the final layer. 




