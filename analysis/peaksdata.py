from utils.distributions import *
from utils.coords import *

# add extra columns with the metrics to the dataframe
def addExtraColumns(df):

    # add extra columns to the dataframe
    elevation  = feet2m(df['elevation in feet'].values)
    prominence = feet2m(df['prominence in feet'].values)
    saddleElev = elevation - prominence
    dominance  = prominence/elevation
    isolation  = df['isolation in km'].values

    # the dataset contains 0.1 for all those unknown isolation peaks (< 1km)
    isolation[isolation <= 0.1] = np.random.uniform(0.05, 0.9, size=(isolation[isolation <= 0.1].size))
    isolation  = isolation

    dsLat = deg2km(df['key saddle latitude'].values - df['latitude'].values)
    miLat = 0.5*(df['key saddle latitude'].values + df['latitude'].values)
    dsLon = deg2km(df['key saddle longitude'].values - df['longitude'].values, latitude=miLat)
    saddleDir = np.mod(np.degrees(np.arctan2(dsLat, dsLon)) + 360, 360)
    saddleDist = np.sqrt(dsLat**2 + dsLon**2)

    dsLat = deg2km(df['isolation latitude'].values - df['latitude'].values)
    miLat = 0.5*(df['isolation latitude'].values + df['latitude'].values)
    dsLon = deg2km(df['isolation longitude'].values - df['longitude'].values, latitude=miLat)
    isolationDir = np.mod(np.degrees(np.arctan2(dsLat, dsLon)) + 360, 360)

    df['elev']       = elevation
    df['prom']       = prominence
    df['dom']        = dominance
    df['isolation']  = isolation
    df['isolDir']    = isolationDir
    df['saddleDir']  = saddleDir
    df['saddleDist'] = saddleDist
    df['saddleElev'] = saddleElev

    return df



def filterPeaksHaversineDist(peaks, diskCenter, diskRadius):
    
    # prefilter peaks
    filterLat = np.logical_and(peaks['latitude'] > diskCenter[0] - km2deg(diskRadius), 
                               peaks['latitude'] < diskCenter[0] + km2deg(diskRadius))
    filterLon = np.logical_and(peaks['longitude'] > diskCenter[1] - km2deg(diskRadius, diskCenter[0]),
                               peaks['longitude'] < diskCenter[1] + km2deg(diskRadius, diskCenter[0]))
    peaksRect = peaks[np.logical_and(filterLat, filterLon)]

    # keep the peaks inside radius
    peaksIdx = []
    for i in range(peaksRect.shape[0]):
        if distance_haversine(diskCenter, (peaksRect.iloc[i]['latitude'], peaksRect.iloc[i]['longitude'])) < diskRadius:
            peaksIdx.append(i)
            
    return peaksRect.iloc[peaksIdx]



def computeDistributions(peaks, diskRadius, detailed=True):
    
    # retrieve values
    elevation    = peaks['elev']
    prominence   = peaks['prom']
    dominance    = peaks['dom']
    isolation    = peaks['isolation']
    isolationDir = peaks['isolDir']
    saddleDir    = peaks['saddleDir']
    saddleDist   = peaks['saddleDist']
    saddleElev   = peaks['saddleElev']
    isolation    = np.minimum(isolation, diskRadius)
    saddleDist   = np.minimum(saddleDist, diskRadius)
    allElev      = np.concatenate([elevation, saddleElev])
    elevRel      = (elevation - elevation.min())/(elevation.max() - elevation.min())
    promRel      = prominence/prominence.max()
    relevance    = np.sqrt(elevRel*promRel)

    # histogram bins
    if detailed:
        # detailed, for synthesis, double the bins of classification so we better target distributions
        percHistogramBins = np.append(np.arange(0, 1.0, 0.025), 1.0)
        anglHistogramBins = np.append(np.arange(0, 360, 30), 360)
        elevHistogramBins = np.append(np.arange(0, 8000, 100), [8000, 8200, 8400, 8600, 8800])
        promHistogramBins = np.append(np.arange(0, 2000, 50), [2000, 2100])
        domiHistogramBins = np.array([0, 0.0035, 0.007, 0.014, 0.021, 0.042, 0.07, 0.105, 0.14, 0.245, 0.49, 0.882, 1.0])
        isolHistogramBins = np.append(np.arange(0, 1.0, 0.0125), 1.0)*diskRadius
    else:
        # bins used in classifier
        percHistogramBins = np.append(np.arange(0, 1.0, 0.05), 1.0)
        anglHistogramBins = np.append(np.arange(0, 360, 30), 360)
        elevHistogramBins = np.append(np.arange(0, 8000, 250), [8000, 8250])
        promHistogramBins = np.append(np.arange(0, 2000, 100), [2000, 2200])
        domiHistogramBins = np.array([0, 0.0035, 0.007, 0.014, 0.021, 0.042, 0.07, 0.105, 0.14, 0.245, 0.49, 0.882, 1.0])
        isolHistogramBins = np.append(np.arange(0, 1.0, 0.05), 1.0)*diskRadius
        
    # histograms
    h_elev       = histogramFromBins(elevation,   bins = elevHistogramBins)
    h_prom       = histogramFromBins(prominence,  bins = promHistogramBins)
    h_dominance  = histogramFromBins(dominance,   bins = percHistogramBins)
    h_groupDomi  = histogramFromBins(dominance,   bins = domiHistogramBins)
    h_saddleDir  = histogramFromBins(saddleDir,   bins = anglHistogramBins)
    h_saddleDist = histogramFromBins(saddleDist,  bins = isolHistogramBins)
    h_saddleElev = histogramFromBins(saddleElev,  bins = elevHistogramBins)
    h_isolation  = histogramFromBins(isolation,   bins = isolHistogramBins)
    h_isolDir    = histogramFromBins(isolationDir,bins = anglHistogramBins)
    h_allElev    = histogramFromBins(allElev,     bins = elevHistogramBins)
    h_elevRel    = histogramFromBins(elevRel,     bins = percHistogramBins)
    h_promRel    = histogramFromBins(promRel,     bins = percHistogramBins)
    h_relevance  = histogramFromBins(relevance,   bins = percHistogramBins)

    # PDF
    p_elev       = h_elev/np.sum(h_elev)
    p_prom       = h_prom/np.sum(h_prom)
    p_dominance  = h_dominance/np.sum(h_dominance)
    p_groupDomi  = h_groupDomi/np.sum(h_groupDomi)
    p_saddleDir  = h_saddleDir/np.sum(h_saddleDir)
    p_saddleDist = h_saddleDist/np.sum(h_saddleDist)
    p_saddleElev = h_saddleElev/np.sum(h_saddleElev)
    p_isolation  = h_isolation/np.sum(h_isolation)
    p_isolDir    = h_isolDir/np.sum(h_isolDir)
    p_allElev    = h_allElev/np.sum(h_allElev)
    p_elevRel    = h_elevRel/np.sum(h_elevRel)
    p_promRel    = h_promRel/np.sum(h_promRel)
    p_relevance  = h_relevance/np.sum(h_relevance)

    # histogram bin midpoints
    x_elev       = (elevHistogramBins[1:] + elevHistogramBins[:-1])/2
    x_prom       = (promHistogramBins[1:] + promHistogramBins[:-1])/2
    x_dominance  = (percHistogramBins[1:] + percHistogramBins[:-1])/2
    x_groupDomi  = (domiHistogramBins[1:] + domiHistogramBins[:-1])/2
    x_saddleDir  = (anglHistogramBins[1:] + anglHistogramBins[:-1])/2
    x_saddleDist = (isolHistogramBins[1:] + isolHistogramBins[:-1])/2
    x_saddleElev = (elevHistogramBins[1:] + elevHistogramBins[:-1])/2
    x_isolation  = (isolHistogramBins[1:] + isolHistogramBins[:-1])/2
    x_isolDir    = (anglHistogramBins[1:] + anglHistogramBins[:-1])/2
    x_allElev    = (elevHistogramBins[1:] + elevHistogramBins[:-1])/2
    x_elevRel    = (percHistogramBins[1:] + percHistogramBins[:-1])/2
    x_promRel    = (percHistogramBins[1:] + percHistogramBins[:-1])/2
    x_relevance  = (percHistogramBins[1:] + percHistogramBins[:-1])/2

    distributions = {
        'elevation' : {'hist': h_elev,       'pdf': p_elev,       'bins': elevHistogramBins, 'x': x_elev },
        'prominence': {'hist': h_prom,       'pdf': p_prom,       'bins': promHistogramBins, 'x': x_prom },
        'dominance' : {'hist': h_dominance,  'pdf': p_dominance,  'bins': percHistogramBins, 'x': x_dominance },
        'domGroup'  : {'hist': h_groupDomi,  'pdf': p_groupDomi,  'bins': domiHistogramBins, 'x': x_groupDomi },
        'saddleDir' : {'hist': h_saddleDir,  'pdf': p_saddleDir,  'bins': anglHistogramBins, 'x': x_saddleDir },
        'saddleDist': {'hist': h_saddleDist, 'pdf': p_saddleDist, 'bins': isolHistogramBins, 'x': x_saddleDist },
        'saddleElev': {'hist': h_saddleElev, 'pdf': p_saddleElev, 'bins': elevHistogramBins, 'x': x_saddleElev },
        'isolation' : {'hist': h_isolation,  'pdf': p_isolation,  'bins': isolHistogramBins, 'x': x_isolation },
        'isolDir'   : {'hist': h_isolDir,    'pdf': p_isolDir,    'bins': anglHistogramBins, 'x': x_isolDir },
        'allElev'   : {'hist': h_allElev,    'pdf': p_allElev,    'bins': elevHistogramBins, 'x': x_allElev },
        'elevRel'   : {'hist': h_elevRel,    'pdf': p_elevRel,    'bins': percHistogramBins, 'x': x_elevRel },
        'promRel'   : {'hist': h_promRel,    'pdf': p_promRel,    'bins': percHistogramBins, 'x': x_promRel },
        'relevance' : {'hist': h_relevance,  'pdf': p_relevance,  'bins': percHistogramBins, 'x': x_relevance }
    }
    return distributions
