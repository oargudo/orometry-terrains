import numpy as np
from scipy.stats import gaussian_kde


def histogram(values, hmin=0.0, hmax=1.0, binSize=0.1, frequencies=False):
    bins = np.append(np.arange(hmin, hmax, binSize), hmax)    
    h = np.histogram(np.clip(values, a_min=None, a_max=bins[-1]), bins=bins)[0]
    if frequencies:
        return h/np.sum(h)
    else:
        return h

def histogramFromBins(values, bins, frequencies=False):
    h = np.histogram(np.clip(values, a_min=None, a_max=bins[-1]), bins=bins)[0]
    if frequencies:
        return h/np.sum(h)
    else:
        return h

def cumulativeDistribution(distribution, inverse=False):
    c = np.cumsum(distribution)
    c = c/c[-1]
    if inverse:
        return 1-c
    else:
        return c

def gaussianKDE(values, bins):
    kde = gaussian_kde(values, bw_method='scott')
    pts = bins[:-1] + 0.5*np.diff(bins)
    return kde.evaluate(pts), pts, kde


def gaussian(x, mu, var):
    return np.exp(-0.5*(x - mu)**2/var)

def gaussian2d(x, y, muX, muY, varX, varY):
    return np.outer(gaussian(x, muX, varX), gaussian(y, muY, varY))
    

def sampleFromPDF(numSamples, pdf, bins):
    # compute CDF
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    # from uniform to CDF bins 
    values = np.random.uniform(size=numSamples)
    value_bins = np.searchsorted(cdf, values)
    # convert bins to value in each bin range
    binsLow = bins
    binsWidth = np.diff(bins)
    random_low = binsLow[value_bins]
    random_off = binsWidth[value_bins]*np.random.uniform(size=numSamples)
    return random_low + random_off

def sampleFromPDF2(numSamples, pdf, binsX, binsY):
    # compute CDF
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    # from uniform to CDF bins 
    values = np.random.uniform(size=numSamples)
    value_bins = np.searchsorted(cdf, values)
    value_bins = np.unravel_index(value_bins, pdf.shape)
    # convert bins to value in each bin range
    xbinsLow = binsX
    xbinsWidth = np.diff(binsX)
    xrandom_low = xbinsLow[value_bins[0]]    
    xrandom_off = xbinsWidth[value_bins[0]]*np.random.uniform(size=numSamples)
    ybinsLow = binsY
    ybinsWidth = np.diff(binsY)
    yrandom_low = ybinsLow[value_bins[1]]    
    yrandom_off = ybinsWidth[value_bins[1]]*np.random.uniform(size=numSamples)    
    # concat coords
    return np.vstack([xrandom_low + xrandom_off, yrandom_low + yrandom_off]).T

def mapToPDF(values, pdf, bins):
    # compute CDF
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    # from uniform to CDF bins 
    value_bins = np.searchsorted(cdf, values)
    value_frac = (values - cdf[value_bins-1])/(cdf[value_bins] - cdf[value_bins-1])
    # convert bins to value in each bin range
    binsLow = bins
    binsWidth = np.diff(bins)
    valLow = binsLow[value_bins] 
    valOff = binsWidth[value_bins]*value_frac
    return valLow + valOff

def evalPDF(values, pdf, bins, interp=True):
    if not interp:
        # find the bin in which the values lie
        return pdf[np.searchsorted(bins, values)-1]
    else:
        return np.interp(values, bins, pdf)

def equalize(data, numBins = 100):
    # assume data already in 0..1
    hist, bins = np.histogram(data, numBins)
    cdf = hist.cumsum()
    cdf = cdf/cdf[-1]
    dataEq = np.interp(data.flatten(), bins[:-1], cdf)
    return dataEq.reshape(data.shape)
    
