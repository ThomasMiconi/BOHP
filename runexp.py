# Submit jobs to the cluster. 

# /opt/python-2.7.10/bin/python


import sys
import os
import shutil

allopts = [


       # With RMSprop, you want MAXDW to be ~ 3 times ETA or more (5 times means clipping will be extremely rare after a while)
       # Without RMSprop (simple SGD), the steps have different sizes for wxy and alpha.. With a same ETA, easier to hit MAXDW with wxy than with alpha...
       # Use RMSprop throughout!


        # For the two-layer, uncorelated-stimuli case

        "ETA 0.01 MAXDW 0.03 WPEN 1e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        
        "ETA 0.001 MAXDW 0.003 WPEN 3e-5 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        "ETA 0.003 MAXDW 0.01 WPEN 1e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        "ETA 0.01 MAXDW 0.03 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        
        "ETA 0.001 MAXDW 0.003 WPEN 1e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        "ETA 0.003 MAXDW 0.01 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        "ETA 0.01 MAXDW 0.03 WPEN 1e-3 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        
        "ETA 0.001 MAXDW 0.003 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        "ETA 0.003 MAXDW 0.01 WPEN 1e-3 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",
        "ETA 0.01 MAXDW 0.03 WPEN 3e-3 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95",


        # For the single-layer, mutually-exclusive-stimuli case. Note that the single-layer network is emulated with a two-layer network with only one neuron in the 'y' (hidden) layer.
        "ETA 0.01 MAXDW 0.03 WPEN 1e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        
        "ETA 0.001 MAXDW 0.003 WPEN 3e-5 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        "ETA 0.003 MAXDW 0.01 WPEN 1e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        "ETA 0.01 MAXDW 0.03 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        
        "ETA 0.001 MAXDW 0.003 WPEN 1e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        "ETA 0.003 MAXDW 0.01 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        "ETA 0.01 MAXDW 0.03 WPEN 1e-3 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        
        "ETA 0.001 MAXDW 0.003 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        "ETA 0.003 MAXDW 0.01 WPEN 1e-3 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        "ETA 0.01 MAXDW 0.03 WPEN 3e-3 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR EXCLUSIVE YSIZE 1 LEARNPERIOD 0 ALPHATRACE .95",
        
       
        ]


for optionz in allopts:

    #dirname = "trial-ref-" + optionz.replace(' ', '-')
    #dirname = "trial-fixedsize-CMN-" + optionz.replace(' ', '-')
    dirname = "trial-logistic-withz-" + optionz.replace(' ', '-')

    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.chdir(dirname)
    print os.getcwd()

    for v in range(20):
        os.mkdir("v"+str(v))
        os.chdir("v"+str(v))
        CMD = "bsub -q short -W 8:00 -eo e.txt -g /rnn /opt/python-2.7.10/bin/python ../../boph.py " + optionz + " RNGSEED " + str(v)
        #CMD = "bsub -q short -W 4:00 -eo e.txt -oo o.txt -g /rnn /opt/python-2.7.10/bin/python ../../rnn.py " + optionz + " RNGSEED " + str(v)
        #CMD = "bsub -q short -W 6:00 -eo e.txt -oo o.txt -g /rnn /opt/python-2.7.10/bin/python ../../min-char-rnn-param.py " + optionz + " RNGSEED " + str(v) # For fixed-size
        #print CMD
        retval = os.system(CMD)
        print retval
        os.chdir('..') 
    
    os.chdir('..') 


    #print dirname
    #for RNGSEED in range(2):
    #st = "python rnn.py COEFFMULTIPNORM " + str(CMN) + " DELETIONTHRESHOLD " + str(DT) + " MINMULTIP " \
    #+ str(MMmultiplierofDT*DT) + " PROBADEL " + str(PD) + " PROBAADD " + str(PAmultiplierofPD * PD) \
    #+ " RNGSEED " + str(RNGSEED) + " NUMBERMARGIN " + str(NM)




