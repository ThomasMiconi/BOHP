# Ref: trial-logistic-withz-ETA-0.003-MAXDW-0.01-WPEN-1e-4-UPDATETYPE-RMSPROP-POSWEIGHTS-0-STIMCORR-UNCORRELATED-YSIZE-2-LEARNPERIOD-20-ALPHATRACE-.95

# This is the code for backpropagation of Hebbian plasticity.
# NOTE: This code is built upon Andrej Karpathy's min-char-rnn.py (https://gist.github.com/karpathy/d4dee566867f8291f086). Any bugs and inefficiencies are entirely my own.
# Requires only numpy.

# This code simulates a simple fear conditioning task: the network must learn
# to determine which of two stimuli is associated with a "pain" signal, and to
# produce output 1 whenever that stimulus is present (even in the absence of
# the pain signal). Which of the two possible stimuli is associated with the
# pain signal changes unpredictably from one episode to the next, but remains
# stable within each episode. The algorithm optimizes both the weight and the
# plasticity of all connections, so that during each episode the network
# quickly learns the necessary associations. See Arxiv preprint for details.

# Default parameters are the ones used in the paper for the 'uncorrelated' problem. You can run the script as-is, or modify some parameters with command-line arguments.
# Examples (from IPython):
# %run bohp.py
# %run bohp.py LEARNPERIOD 0 YSIZE 3
# %run bohp.py ETA 0.003 MAXDW 0.01 WPEN 3e-4 UPDATETYPE RMSPROP POSWEIGHTS 0 STIMCORR UNCORRELATED YSIZE 2 LEARNPERIOD 20 ALPHATRACE .95

# The program will store the mean absolute errors of each episode in errs.txt, as well as other data (weights, etc.) in data.pkl. See code.

# The only difference with the paper is that we now use a logistic (strictly-positive) output for the final layer. 

# Note that this program contains a lot of commented-out code for gradient checking. 

import  numpy as np
import sys
import cPickle as pickle

#np.set_printoptions(suppress=True, precision=8)  

xsize=3
ysize=2  # Can be modified by command-line argument
zsize=1  # Don't change this unless you change the rest..

g = {
'NBITER': 100,      # 20 for gradient checking
'YSIZE': 2,         # Number of neurons in the y layer (note: if 1, this is essentially equivalent to a single-layer network since the last layer is a single one-to-one connection)
'PROBAPAIN': .3,    # Probability of delivering a 'pain' signal when the pain-associated stimulus is present
'LEARNPERIOD': 20,  # Exploratory 'learning period' at the beginning of each episode, during which we do not keep track of / try to descend the error. Not strictly needed, but improves performance.
'POSWEIGHTS': 0,    # If you want to enforce all weights to be positive (don't!)
'UPDATETYPE': 'RMSPROP',
'ALPHATRACE' : .95, # The Gamma parameter in the paper (time constant for the exponential decay of the running average of input-output products fot the Hebbian traces)
'NBTRIALS': 20000,  # Total number of episodes. For gradient checking, always 3. 
'RNGSEED' : 0,      # Random seed!
'ETA' : .003,       # Learning rate
'MAXDW' : .01,      # Maximum absolute value of parameter modification after every episode (see below)
'STIMCORR': 'UNCORRELATED', # Whether the two stimuli are 'EXCLUSIVE' or 'UNCORRELATED'. The former is much easier.
'WPEN' : 1e-4,      # Coefficient of the L1 penalty on weights for regularization.
'TEST': 0           # If you want to run using weights from a saved file (see below)
}

'''
FROZENINPUTSIZE = 500; frozeninputs = .2 * np.random.randn(xsize, FROZENINPUTSIZE)  # For debugging / gradient checking
WINC = 0.003 *  np.random.randn(ysize, xsize)
ALPHAINC = 0.003 * np.random.randn(ysize, xsize)
WINC.fill(.005); ALPHAINC.fill(.005)
'''


# Parsing command line arguments
argpairs = [sys.argv[i:i+2] for i in range(1, len(sys.argv), 2)]
for argpair in argpairs:
    if not (argpair[0] in g):
        raise ValueError("Error, tried to pass value of non-existent parameter "+argpair[0])
    if (argpair[0] == 'UPDATETYPE') or (argpair[0] == 'STIMCORR') :  # String (non-numeric) parameters
        g[argpair[0]] = argpair[1]
    else:
        g[argpair[0]] = float(argpair[1])

# Initialization!
ysize = int(g['YSIZE'])
np.random.seed(int(g['RNGSEED']))
wxy = np.random.randn(ysize, xsize) * .1
by = np.random.randn(ysize) * .1
wyz =  np.abs(np.random.randn(zsize, ysize) * .1)
bz = np.random.randn(zsize) * .1
alpha = np.random.randn(ysize, xsize) * .1
hebb = np.zeros((ysize, xsize)) 
mwxy, mwyz, malpha , mbz, mby = np.zeros_like(wxy), np.zeros_like(wyz), np.zeros_like(alpha), np.zeros_like(bz), np.zeros_like(by)



archy = []
archz = []
errs=[]
archerrs = []

if g['TEST']:
    testdir='trial-corr-withz-ETA-0.01-MAXDW-0.03-WPEN-3e-4-UPDATETYPE-RMSPROP-POSWEIGHTS-0-STIMCORR-UNCORRELATED-YSIZE-2-LEARNPERIOD-0-ALPHATRACE-.95/v8/'
    with open(testdir+'data.pkl', 'r') as handle:
              (wxy, wyz, alpha, by, bz, hebb, errs, g) = pickle.load(handle)
    g['TEST'] = 1  # because the data file will obliterate g



for trial in range(g['NBTRIALS']):
    
    # For gradient checking:
    #wxy += WINC
    #alpha += ALPHAINC
    
    # We alternate which of the two stimuli is associated with potential pain delivery   
    PAINSTIM = 1 + trial % 2
    #PAINSTIM = 1  # For debugging
    dydas=[]
    dydws=[]
    xs , ys, zs = [], [], []
    hebb.fill(0)

    # Run the episode!
    for n in range(g['NBITER']):
        #x = .25 * np.random.randn(xsize, 1)
        #x.fill(.25)
        
        x = (np.random.rand(xsize, 1) < .5).astype(int)
        if g['STIMCORR'] == 'EXCLUSIVE':
            x[2] = 1 - x[1]
        elif g['STIMCORR'] == 'UNCORRELATED':
            pass  # NO-OP
        else:
            raise ValueError("Which correlation between stimuli?")

        x[0] = 0
        if (x[PAINSTIM] > 0.9) and (np.random.rand() < g['PROBAPAIN']):
            x[0] = 1

        #x = frozeninputs[:, n % FROZENINPUTSIZE, None]  # For debugging / gradient-checking
        #y = np.dot(wxy, x) + np.dot (alpha*hebb, x)  # Linear output, for debugging 
        #z = np.dot(wyz, y)
        

        y = np.tanh(np.dot(wxy, x) + np.dot (alpha*hebb, x) + by[:, None])   # Tanh nonlinearity on the y
        #z = np.tanh(np.dot(wyz, y) + bz)   # Tanh nonlinearity on the z
        z = 1.0 / (1.0 + np.exp(-(np.dot(wyz, y) + bz)))  # Logistic nonlinearity on the z 
        

        # Okay, now we comnpute the quantities necessary for the actual BOHP algorithm.

        # Pre-compute the gammas for the gradients of the Hebbian traces (see paper). Note that n is the number of steps already done.
        gammas = [(1.0 - g['ALPHATRACE'])*(g['ALPHATRACE']**(n-(xx+1))) for xx in range(n)]  

        # Gradients of y(t) wrt alphas (dyda):
        summand=0
        for k in range(xsize):
            summand += alpha[None, :, k].T * x[k] * sum([prevx[k] * prevdy *gm for prevx, prevdy, gm in zip(xs, dydas, gammas)]) # Gradients of the Hebbian traces wrt the alphas
        dyda = summand + x.T * hebb   # Note that dyda is a matrix, of the same shape as wxy (there is one gradient per alph, and one alpha per connection)

        # Gradients of y(t) wrt ws (dydw):
        summand=0
        for k in range(xsize):
            summand += alpha[None, :, k].T * x[k] * sum([prevx[k] * prevdy *gm for prevx, prevdy, gm in zip(xs, dydws, gammas)]) 
        dydw = summand + x.T * np.ones_like(wxy)

        # Store relevant variables of this timestep (x, y, z, gradients)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        dydas.append(dyda)
        dydws.append(dydw)

        # Update the Hebbian traces (remember that 'hebb' is a matrix of same dimensionality as wxyi - one hebbian trace per connection)
        hebb += (1.0 - g['ALPHATRACE']) * (np.dot(y, x.T) - hebb)

        # End of the episode!

    # Archive the ys of this episode for debugging
    archy.append(np.array(ys))
    
    #if trial == 1:   # For gradient checking
    #    rdydas = np.array(dydas).copy()
    #    rdydws = np.array(dydws).copy()
    

    # Now that the episode is completed, we can run some backpropagation. 

    # First, let us compute the error at each timestep in the topmost layer (z)
    # The target output for each timestep is the value of the x associated with pain:
    tgts = np.array(xs)[:, PAINSTIM, 0]
    zsflat = np.array(zs).flatten()   # Flattens the z's of this episode into a single vector - NOTE: assumes zsize=1
    archz.append(zsflat)

    # Mean-squared error
    errors = (zsflat - tgts) ** 2
    errors[:g['LEARNPERIOD']].fill(0)  # We don't care about early trial, exploratory "learning" period.....  Not strictly needed, but makes things a bit better.
    archerrs.append(errors)
    # if trial  == 1 :  # For gradient checking, you use 3 trials and you only compute the gradients in the middle one
    

    # Now we compute the gradient of the error wrt z at each timestep:
    dedzs = 2*(zsflat -tgts).copy()             # Derivative of squared error
    dedzsraw = dedzs * zsflat * (1 - zsflat)    # Gradient through logistic nonlinearity
    #dedzsraw = dedzs * (1 - zsflat * zsflat)   # Gradient through tanh nonlinearity
    #dedzsraw = dedzs.copy() # Gradient through linear output, for debugging
    #dedzsraw = zsflat - tgts   # Experimental hack attempt at cross-entropy-ish error. Didn't seem to make much difference
    

    # We backpropagate this through the yz weights to get the gradient of the error wrt the y's at each timestep:
    dedys =  np.dot(wyz.T, dedzsraw[None, :])

    ysflat = np.array(ys)[:,:,0].T
    dedysraw = dedys * (1 -  ysflat * ysflat)   # Gradient through tanh nonlinearity
    #dedysraw = dedys.copy() 

    '''
# This checks the gradient on ys 
print "Predicting change in y:"
diff = np.array(np.arctanh(archy[2])) - np.array(np.arctanh(archy[0])) 
#diff = np.array(archy[1]) - np.array(archy[0]) 
calcdiffa = np.sum(rdydas*ALPHAINC*2, axis=2)[:,:,None]
calcdiffw = np.sum(rdydws*WINC*2, axis=2)[:,:,None]
calcdifftot = calcdiffa + calcdiffw
print " Measured diff.:"
print diff.T
print "Calculated diff.:"
print calcdifftot.T
print "Error:"
print diff.T-calcdifftot.T

print "Predicting change in final error (in z):"
differr = archerrs[2] - archerrs[0]  
diffz = archz[2] - archz[0]  
diffy = archy[2] - archy[0]; diffy = diffy[:,:,0]
predicteddifferrfromdiffy = np.sum(dedys * diffy.T, axis=0)    
dedas = dedysraw.T[:, :, None] * rdydas
dedws = dedysraw.T[:, :, None] * rdydws
calcdifferr_a = np.sum(np.sum(dedas*ALPHAINC*2, axis=2), axis=1)
calcdifferr_w = np.sum(np.sum(dedws*WINC*2, axis=2), axis=1)
calcdifferr_tot = calcdifferr_a + calcdifferr_w
print "Error:"
print differr - calcdifferr_tot
    '''


    # Now we use the computed gradients for actual weight modification, using the quantities computed at each timestep during the episode!
    if not g['TEST']:
        # First, the biases by and bz:
        dbz = np.sum(dedzsraw, axis=0)    
        dby = np.sum(dedysraw, axis=1)
        # Then the fixed weight matrices wxy and wyz, and the alpha's:
        dwyz = np.dot(ysflat, dedzsraw[:, None]).T
        dalpha = np.sum(dedysraw.T[:, :, None] * np.array(dydas), axis=0)
        dwxy =  np.sum(dedysraw.T[:, :, None] * np.array(dydws), axis=0)
        
        for param, dparam, mem in zip([wxy, wyz, alpha, by, bz], 
                                    [dwxy, dwyz, dalpha, dby, dbz],
                                    [mwxy, mwyz, malpha, mby, mbz] ):
            #mem = .99 * mem + .01 * dparam * dparam   # Does NOT update the m-variables - head.desk()
            if trial == 0:
                mem +=  dparam * dparam 
            else:
                mem +=  .01 * (dparam * dparam - mem)    # Does update the m-variables
            RMSdelta = -g['ETA'] * dparam / np.sqrt(mem + 1e-8) 
            #print np.mean( np.abs(RMSdelta) >  g['MAXDW'])
            np.clip(RMSdelta, -g['MAXDW'], g['MAXDW'], out = RMSdelta)
            
            if g['UPDATETYPE'] == 'RMSPROP':   # RMSprop update
                delta = dparam / np.sqrt(mem + 1e-8) 
            elif g['UPDATETYPE'] == 'SIMPLE':  # Simple SGD
                delta = dparam
            else:
                raise ValueError('Wrong / absent update type!')
            
            # Notice the clipping
            param += np.clip( -g['ETA'] * delta , -g['MAXDW'], g['MAXDW'])   
            #print np.mean(np.abs(g['ETA'] * delta) > g['MAXDW'])


        #alpha += np.clip(g['ETA'] * np.sum(dedysraw.T[:, :, None] * np.array(dydas), axis=0), -g['MAXDW'], g['MAXDW'])
        ##print np.mean(np.abs(g['ETA'] * np.sum(dedysraw.T[:, :, None] * np.array(dydas), axis=0))>g['MAXDW'])
        #wxy += np.clip(g['ETA'] * np.sum(dedysraw.T[:, :, None] * np.array(dydws), axis=0), -g['MAXDW'], g['MAXDW'])

        #  L1-penalty on all weights and alpha, for regularization (really useful!)
        alpha -= g['WPEN'] * np.sign(alpha)
        wxy -= g['WPEN'] * np.sign(wxy)
        wyz -= g['WPEN'] * np.sign(wyz)

        if g['POSWEIGHTS']:   # Just don't!
            wxy[wxy<1e-6] = 1e-6
            wyz[wyz<1e-6] = 1e-6
            alpha[alpha<1e-6] = 1e-6

    # Log the total error for this episode
    meanerror = np.mean(np.abs(errors[g['LEARNPERIOD']:]))

    # Every 10th trial, display a message and update the output file
    if trial % 10 == 0:
        print "Episode #", trial, " - mean abs. error per timestep (excluding learning period): ", meanerror
        print "Errors at each timestep (should be 0.0 for early learning period):"
        print errors
        np.savetxt("errs.txt", np.array(errs))
        with open('data.pkl', 'wb') as handle:
              pickle.dump((wxy, wyz, alpha, by, bz, hebb, errs, g), handle)
        ## For reading:
        #with open('data.pkl', 'r') as handle:
        #      (wxy, wyz, alpha, by, bz, hebb, errs, g) = pickle.load(handle)
    errs.append(meanerror)

