import numpy as np

from theano import tensor as T

def logistic(x):
    return 1 / (1 + np.exp(-x))

def logit(x):
    return np.log(x) - np.log(1-x)

def logisticT(x):
    return 1 / (1 + T.exp(-x))

def logitT(x):
    return T.log(x) - T.log(1-x)

def betaln(a, b):
    return T.gammaln(a) + T.gammaln(b) - T.gammaln(a + b)

def betapriorln(x, a, b):
    return (a-1.)*T.log(x) + (b-1.)*T.log(1.-x) - betaln(a, b)
