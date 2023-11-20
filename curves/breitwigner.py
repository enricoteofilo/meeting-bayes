import numpy as np
from scipy.special import logsumexp
import os
import psutil
import raynest.model

#defining a uniform prior (in log) which is the most agnostic choice
def FindHeightForLevel(inArr, adLevels):
    """
    Computes the height of a :math:`2D` function for given levels
    
    :param inArr: function values
    :type inArr: array
    :param adLevels: levels
    :type adLevels: list or array
    
    :return: function values with levels closest to *levels*
    :rtype: array
    """
    
    # flatten the array
    oldshape = np.shape(inArr)
    adInput  = np.reshape(inArr,oldshape[0]*oldshape[1])
    
    # get array specifics
    nLength  = np.size(adInput)

    # create reversed sorted list
    adTemp   = -1.0 * adInput
    adSorted = np.sort(adTemp)
    adSorted = -1.0 * adSorted

    # create the normalised cumulative distribution
    adCum    = np.zeros(nLength)
    adCum[0] = adSorted[0]
    
    for i in range(1,nLength):
        adCum[i] = np.logaddexp(adCum[i-1], adSorted[i])
        
    adCum    = adCum - adCum[-1]

    # find the values closest to levels
    adHeights = []
    for item in adLevels:
        idx = (np.abs(adCum-np.log(item))).argmin()
        adHeights.append(adSorted[idx])

    adHeights = np.array(adHeights)
    return np.sort(adHeights)

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

def line(x,theta):
    return theta[0]*x + theta[1]

def BreitWigner(x, theta):
    #theta[0]=M of resonance; theta[1]=Gamma/M
    ff = (np.sqrt(8)/np.pi)*(1/theta[0])*np.sqrt((1+theta[1]**2)/(1+np.sqrt(1+theta[1]**2)))*theta[1]/((x**2 -1.0)**2 + theta[1]**2)
    return ff

def data_gen(x,model,theta,rng,sigma):
    #impongo quale sia il generatore delle incertezze
    if rng is None:
        rng=np.random
    noise = rng.normal(0.0, sigma, size=(x.shape[0]))
    signal = model(x,theta)
    return noise + signal

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    rng = np.random.default_rng(4269420)
    M = 1.0e6
    gamma = 5.0e5
    theta = (M,gamma/M)

    x = np.linspace(0.0,2.0e0*(np.sqrt(1+theta[1]**2)),100)
    yy = BreitWigner(x,theta)
    sigma_noise=np.max(yy)/2.0
    y = data_gen(x,BreitWigner,theta,rng=rng,sigma=sigma_noise)
    
    #what are we printing here? Is it the likelihood computed over the known parameters but given the data with uncertanties as an input?
    print('logL (simulation)= ', log_likelihood(x,y,line, theta,sigma=sigma_noise))
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    fig = plt.figure(1)
    plt.title('Bayesian inference on Breit Wigner')
    plt.plot(x,yy, linestyle='--', color='#7393B3', zorder=1,label='true')
    plt.errorbar(x,y, yerr=sigma_noise, xerr=None, marker='.', linestyle=' ', color=(0,0,0), zorder=2, label='sim w\ Gaussian noise')
    ax.set_xlabel('x')
    ax.set_ylabel('cross section')
    plt.legend(loc='best')
    plt.show()
    
    
    nbins = 256
    
    '''
    ##JUST A LINE
    #we're defining the parameter space on which we want to do the inference
    intercept = np.linspace(1.0e-7,1.2e-6,nbins)
    slope = np.linspace(-1.0e-6,0.5e-6,nbins)
    #check later if the posterior rails against the prior
    logP = np.zeros((nbins,nbins),np.float64)
    #now we compute the actual logP over the entire parameter space
    for i in range(nbins):
        for j in range(nbins):
            logP[i,j] = log_posterior(x,y,line,(slope[i],intercept[j]),sigma=sigma_noise)
    
    #now we print the <insert name here> necessary for odds ratio calculation
    print('logZ (line)= {}'.format(logsumexp(logP)*np.diff(x)[0]*np.diff(y)[0]) )
    #we compute the contours for given probability regions
    #levels = np.sort(FindHeightForLevel(logP,[0.5,0.9,0.99]))
    
    #making the 2D plot of logP on the whole parameter space
    fig = plt.figure(2)
    X,Y = np.meshgrid(slope, intercept)
    ax = fig.add_subplot(121)
    #contours
    #norm = colors.Normalize(vmin=0.0, vmax=1.0)
    C = ax.contourf(X,Y, np.exp(logP.T), 1000, cmap='Blues') #norm=norm)
    ax.contour(X,Y,np.exp(logP.T),np.exp(levels),linestyles='-',colors='k')
    ax.set_xlabel('slope')
    ax.set_ylabel('intercept')
    plt.colorbar(C)
    
    '''
    ##ACTUAL BREIT WIGNER
    fig = plt.figure(3)
    decaywidth = np.linspace(0.0,2.0,nbins)
    mass = np.linspace(1.0,3.0e6,nbins)
    logP2 = np.zeros((nbins,nbins),np.float64)
    #now we compute the actual logP over the entire parameter space
    for i in range(nbins):
        for j in range(nbins):
            logP2[i,j] = log_posterior(x,y,BreitWigner,(mass[i],decaywidth[j]),sigma=sigma_noise)
    #now we print the <insert name here> necessary for odds ratio calculation
    print('logZ (Breit Wigner)= {}'.format(logsumexp(logP2)*np.diff(x)[0]*np.diff(y)[0]) )
    #we compute the contours for given probability regions
    levels2 = np.sort(FindHeightForLevel(logP2,[0.5,0.9,0.99]))

    
    X,Y = np.meshgrid(mass, decaywidth)
    ax = fig.add_subplot(111)
    #contours
    P2max=np.exp(logP2.max())
    norm = colors.Normalize(vmin=0.0, vmax=P2max)
    D = ax.contourf(X,Y, np.exp(logP2.T), 1000, cmap='Blues', norm=norm)
    ax.contour(X,Y,np.exp(logP2.T),np.exp(levels2),linestyles='-',colors='k')
    ax.set_xlabel('mass')
    ax.set_ylabel('decaywidth/mass')
    plt.colorbar(D)
    #plt.show()
    
    