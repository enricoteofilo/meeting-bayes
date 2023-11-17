import numpy as np
from scipy.special import logsumexp
import os
import psutil
import raynest.model

#defining a uniform prior (in log) which is the most agnostic choice
def log_prior(theta):
    return 0.0

#defining the likelihood
def log_likelihood(x,y,model,theta,sigma):
    prediction = model(x,theta) #what the model predicts given some values for its parameters
    residuals = (y-prediction)/sigma
    #calculating the (logarithm of the) actual likelihood
    logL = -0.5*np.sum(residuals**2)
    return logL

#calculating the posterior from Bayes theorem
def log_posterior(x,y,model,theta,sigma):
    return log_prior(theta)+log_likelihood(x,y,model,theta,sigma=sigma)

def holepot(x, theta):
    return (1.0-np.exp(-(theta[0]+x**3)/(theta[1]*x**2)))/(1+np.sqrt(x))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(4269420)
    theta = (4.0,100.0)

    x = np.linspace(1.0e-2,1.0e2,100000)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    fig = plt.figure(1)
    plt.plot(x,holepot(x,theta))
    ax.set_xlabel('function')
    ax.set_ylabel('x')
    plt.show()