import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.morphology import medial_axis
import cv2 as cv
import ot
import time
from utils.coords import *
from utils.distributions import *
from utils.metrics import *


############
# 1. PEAKS #
############

def computePeakProbability(coords, elev, normCoords, elevDeviation, probMap, elevMap, kdtree, treePoints, treeElevs, distributions):

    # probability map
    pmapCoords = [(normCoords[:,0]*probMap.shape[0]).astype(int), (normCoords[:,1]*probMap.shape[1]).astype(int)]
    peakProb   = probMap[pmapCoords[0], pmapCoords[1]]

    # elevation map term
    if elevMap.size > 0:
        gsigmaElev = elevDeviation
        emapCoords = [(normCoords[:,0]*elevMap.shape[0]).astype(int), (normCoords[:,1]*elevMap.shape[1]).astype(int)]
        mapElev    = elevMap[emapCoords[0], emapCoords[1]]
        peakProb  *= norm(loc=0, scale=gsigmaElev).pdf(elev - mapElev)/norm(loc=0, scale=gsigmaElev).pdf(0)
    
    # or neighboring elevations term
    elif kdtree:
        # TODO: test and tune this alternative peak placement that does not use the elevation map
        peakNeighs = kdtree.query_ball_point(coords, r=5.0)[0]  # 5km
        if len(peakNeighs):
            gsigmaNeighs = elevDeviation
            avgNeighElev = np.mean(treeElevs[peakNeighs])
            peakProb *= norm(loc=0, scale=gsigmaNeighs).pdf(elev - avgNeighElev)/norm(loc=0, scale=gsigmaNeighs).pdf(0)

    # isolation term
    if kdtree:
        isol, iHigher = kdtree.query(coords, k=1)
        visol   = (treePoints[iHigher[0]] - coords).squeeze()
        isolDir = np.mod(np.degrees(np.arctan2(visol[1], visol[0])) + 360, 360)
        
        # 0.5 = ad-hoc factor to account for the fact that isolation is not "peak-to-peak" but "peak-to-higher ground"
        # so, we are matching towards 2*isolation distribution assuming that halfway through this distance we will
        # already find some higher ground. Without using this multiplier, peaks get too clumped together.
        peakProb *= evalPDF(isol*0.5, distributions['isolation']['hist']/distributions['isolation']['hist'].max(), 
                                      distributions['isolation']['bins'][:-1])
        peakProb *= evalPDF(isolDir,  distributions['isolDir']['hist']/distributions['isolDir']['hist'].max(),   
                                      distributions['isolDir']['bins'][:-1])
        # also, note that instead of a true PDF, we used a normalized histogram (divided by its max value)
        # otherwise the probabilities are always too low and peaks end up just being rejected all the time
        
        # avoid placing peaks too close to each other
        # 100m, approx cell resolution for SRTM 3 arc-sec used in the analysis
        if isol < 0.1: 
            peakProb = 0
        
    return peakProb


def generatePeaks(numPeaks, terrainSize, distributions, probMap, elevMap, fixedPeaks, 
                  elevRangeFilter=0.15, elevMapDeviation=0.05, maxTrials=100):

    numSamples  = numPeaks
    synthCoords = np.empty((numSamples, 2))
    synthElevs  = sampleFromPDF(numSamples, distributions['elevation']['hist'], distributions['elevation']['bins']) 

    hmin        = distributions['elevation']['bins'][:-1][distributions['elevation']['hist'] > 0][0]
    hmax        = distributions['elevation']['bins'][:-1][distributions['elevation']['hist'] > 0][-1]
    elevRange   = hmax - hmin
    
    for pi in range(numSamples):
        
        # neighbors kd-tree
        fixedHigher = fixedPeaks[fixedPeaks[:,2] > synthElevs[pi]]
        treePoints  = np.concatenate([fixedHigher[:,:2], synthCoords[:pi,:]])
        treeElevs   = np.concatenate([fixedHigher[:,2],  synthElevs[:pi]]) 
        if treePoints.shape[0] > 0:
            kdtree  = cKDTree(treePoints)
        else:
            kdtree  = None
        
        # possible positions by filtering elevation 
        pminElev = synthElevs[pi] - elevRangeFilter*elevRange
        pmaxElev = synthElevs[pi] + elevRangeFilter*elevRange
        #posCandidates = np.nonzero(np.logical_and(probMap > 0, np.logical_and(elevMap > pminElev, elevMap < pmaxElev)))
        posCandidates = np.nonzero(np.logical_and(elevMap > pminElev, elevMap < pmaxElev))
        posCandidates = np.transpose(posCandidates)
        
        # dart throwing
        numTrials = 0
        placed    = False
        bestProb  = 0
        while numTrials < maxTrials and not placed:
        
            # sample a position
            randIdx = np.random.randint(0, high=posCandidates.shape[0])
            pm = posCandidates[randIdx,:].reshape((1,2)) + np.random.uniform(0, 1, size=(1,2))
            pn = pm/elevMap.shape
            pc = terrainSize * pn
            
            # compute probability
            peakProb = computePeakProbability(pc, synthElevs[pi], pn, elevRange*elevMapDeviation,
                                              probMap, elevMap, kdtree, treePoints, treeElevs, distributions)
            
            # do we accept the peak?
            if np.random.uniform() < peakProb:
                synthCoords[pi,:] = pc
                placed = True
            elif bestProb <= peakProb:
                synthCoords[pi,:] = pc
                bestProb = peakProb
            
            numTrials += 1
    
    
    return synthCoords, synthElevs



#############
# 2. RIDGES #
#############


def delaunaySaddleEdgeWeight(peak1, elev1, saddle, peak2, elev2, probMap, terrainSize, delaunayRidgeExponent):

    # in case we wanted to condition saddle placement to a probability map
    # usually leads to unnatural ridges though, so we do not use it
    pProb = probMap[np.clip((saddle[0]/terrainSize[0])*probMap.shape[0], 0, probMap.shape[0]-1).astype(int), 
                    np.clip((saddle[1]/terrainSize[1])*probMap.shape[1], 0, probMap.shape[1]-1).astype(int)]
    pProb = np.maximum(pProb, 0.000001)

    # peaksDist in km, elev in m
    minPeakElev = np.minimum(elev1, elev2)
    peaksDist  = np.linalg.norm(peak1 - peak2)
    
    return (1 / pProb) * 1000*peaksDist/(minPeakElev**delaunayRidgeExponent)  

    
def createDelaunayGraph(peakCoords, peakElevs, probMap, terrainSize, distributions, delaunayRidgeExponent):
    
    # delaunay
    dtris = Delaunay(peakCoords)
    numTris = dtris.simplices.shape[0]
    
    saddleCoords = np.empty((3*numTris, 2))
    saddlePeaks  = np.empty((3*numTris, 2), dtype=np.int)
    saddleWeight = np.empty((3*numTris, 1))
    numSaddles = 0

    # assign weight to each edge
    for tri in dtris.simplices:
        for ti in range(3):
            i  = tri[ti]
            j  = tri[(ti+1)%3]
            
            midPoint   = 0.5*(peakCoords[i] + peakCoords[j])
            peakDist   = np.linalg.norm(peakCoords[i] - midPoint)
            offset     = np.random.normal(0, 0.25*peakDist) # this nearly all samples (4 sigma) inside radius
            offDir     = np.random.uniform(0, 2*np.pi)
            saddleX    = np.clip(midPoint[0] + offset*np.cos(offDir), 0, terrainSize[0])
            saddleY    = np.clip(midPoint[1] + offset*np.sin(offDir), 0, terrainSize[1])

            saddleCoords[numSaddles] = [saddleX, saddleY]     
            saddlePeaks [numSaddles] = [i, j]
            saddleWeight[numSaddles] = delaunaySaddleEdgeWeight(peakCoords[i], peakElevs[i], 
                                                                np.array([saddleX, saddleY]), 
                                                                peakCoords[j], peakElevs[j],
                                                                probMap, terrainSize,
                                                                delaunayRidgeExponent)    
                                                                
            numSaddles += 1
            
    
    saddleCoords = saddleCoords[:numSaddles,:]
    saddlePeaks  = saddlePeaks [:numSaddles,:]
    saddleWeight = saddleWeight[:numSaddles,:]
    
    return saddleCoords, saddlePeaks, saddleWeight
    
    
def generateRidgeTree(peakCoords, peakElevs, saddleCoords, saddlePeaks, saddleWeight):
    
    numPeaks = peakElevs.size
    
    # construct matrices:
    # G: cost matrix
    # S[i,j] = id of saddle connecting peak i to peak j    
    G = np.zeros((numPeaks, numPeaks))
    S = np.full((numPeaks, numPeaks), -1)
    for i in range(saddleWeight.size):
        p1 = saddlePeaks[i,0]
        p2 = saddlePeaks[i,1]
        # update matrices if no edge yet or better edge
        # note each edge can appear twice but we do not care about directionality
        if S[p1, p2] < 0 or saddleWeight[i] < G[p1, p2]:
            G[p1, p2] = G[p2, p1] = saddleWeight[i] if saddleWeight[i] > 0 else 1e-12
            S[p1, p2] = S[p2, p1] = i
    
    # build tree
    G = csr_matrix(G)
    T = minimum_spanning_tree(G)
 
    # clean saddles to those only in T
    numResSaddles   = numPeaks - 1
    resSaddleCoords = np.zeros((numResSaddles, 2))                   # coords
    resSaddlePeaks  = np.full ((numResSaddles, 2), -1, dtype=np.int) # the two peaks on this saddle
    saddleMaxHeight = np.zeros((numResSaddles, ))                    # maximum possible height (min of both peaks)
    saddleWidth     = np.zeros((numResSaddles, ))                    # ridge width of the saddle

    RidgeTree   = np.full((numPeaks, numPeaks),   -1)
    saddleRemap = np.full((saddleWeight.size, ), -1)
    currSaddle  = 0
    Trow, Tcol  = T.nonzero()
    for i in range(len(Trow)):
        peak1  = Trow[i]
        peak2  = Tcol[i]
        saddle = S[peak1, peak2]
        if saddleRemap[saddle] < 0:
            saddleRemap[saddle] = currSaddle
            resSaddleCoords[currSaddle] = saddleCoords[saddle]
            resSaddlePeaks[currSaddle]  = saddlePeaks[saddle]
            saddleMaxHeight[currSaddle] = np.minimum(peakElevs[peak1], peakElevs[peak2])
            saddleWidth[currSaddle]     = np.linalg.norm(peakCoords[peak1] - saddleCoords[saddle]) \
                                        + np.linalg.norm(peakCoords[peak2] - saddleCoords[saddle])
            RidgeTree[peak1, peak2] = RidgeTree[peak2, peak1] = currSaddle
            currSaddle += 1
        else:
            print('Sanity check, this should never happen!', saddle)

    return RidgeTree, resSaddleCoords, resSaddlePeaks, saddleMaxHeight, saddleWidth



##############
# 3. SADDLES #
##############

def computeCostDominances(peakElevs, peakDoms, peakRangeDom, sampleDoms, peakSaddle, 
                          saddleMaxIncrease, saddleWidth):
    
    numPeaks   = peakElevs.size
    numSamples = sampleDoms.size    

    # idem for the case we lower a peak height, we
    # might potentially affect all the highers
    peaksByHeight = np.argsort(peakElevs)[::-1] + 1
    highestPeak = peakElevs.argmax()
            
    C = np.zeros((numPeaks, numSamples), dtype=np.float)
    for i in range(numPeaks):    
        
        if i == highestPeak:
            C[i,:] = 1e12
            C[i,sampleDoms >= 1] = 0
            continue
                        
        for j in range(numSamples):
            
            # if dominance not in allowed range for this peak, impossible
            if sampleDoms[j] < peakRangeDom[i,0] or sampleDoms[j] > peakRangeDom[i,1]:
                C[i,j] = 1e12
                continue
            
            dDom  = sampleDoms[j] - peakDoms[i]
            dElev = dDom * peakElevs[i]
            
            # lowering saddle we can always get any dom
            # or increasing the saddle up to the maximum height
            if dElev >= 0 or np.abs(dElev) < saddleMaxIncrease[peakSaddle[i]]:
                cost   = np.abs(dElev)**2 / (1000*saddleWidth[peakSaddle[i]])
                C[i,j] = cost

            # otherwise, we need to increase peak height (not desirable)
            else:
                penaltyIncreasePeak = 1e6*numPeaks
                cost   = np.abs(dElev)              
                C[i,j] = 1e12

    return C
    
    
def matchDominances(peakElevs, peakDoms, peakRangeDom, sampledDoms, peakSaddle, 
                    saddleInitElevs, saddleMaxElevs, saddleWidth):

    numPeaks = peakElevs.size
    saddleMaxIncrease = saddleMaxElevs - saddleInitElevs

    # account for highest peak
    if sampledDoms.max() < 1:
        highestPeak = peakElevs.argmax()
        sampledDoms[sampledDoms.argmin()] = 1
    
    # compute cost matrix
    DomCostMatrix = computeCostDominances(peakElevs, peakDoms, peakRangeDom, sampledDoms,  
                                          peakSaddle, saddleMaxIncrease, saddleWidth)
    
    # solver using optimal transport, optimal reached if enough iters
    a = np.ones((numPeaks,))
    b = np.ones((numPeaks,))
    otMatrix = ot.emd(a, b, DomCostMatrix, numItermax=1000000000)
    otDest = otMatrix.argmax(axis=1)
    otCost = DomCostMatrix[np.arange(numPeaks), otDest]
    destDoms = sampledDoms[otDest]
    
    # assign elevations
    saddleElevs = saddleInitElevs
    numChanges  = 0
    for i in range(numPeaks):

        domDst  = sampledDoms[otDest[i]]
        domSrc  = peakDoms[i]
        domDiff = domDst - domSrc
        diffH   = -peakElevs[i]*domDiff # to increase dom, reduce saddle H
        psaddle = peakSaddle[i]
                
        if np.abs(domDiff) < 0.001:
            continue       

        # modify if possible the saddle
        if diffH < saddleMaxIncrease[psaddle]:
            saddleElevs[psaddle] = np.maximum(saddleElevs[psaddle] + diffH, 0)            
            numChanges += 1            
        # otherwise, modify the peak
        #else:
        #    peakElevs[i] += np.abs(diffH)

    return peakElevs, saddleElevs, numChanges
    
    
def computeCostProminences(peakElevs, peakProms, peakRangeProm, sampleProms, 
                           peakSaddle, saddleMaxIncrease, saddleWidth):
    
    numPeaks   = peakElevs.size
    numSamples = sampleProms.size    

    # idem for the case we lower a peak height, we
    # might potentially affect all the highers
    peaksByHeight = np.argsort(peakElevs)[::-1] + 1
    highestPeak = peakElevs.argmax()
    
    C = np.zeros((numPeaks, numSamples), dtype=np.float)
    for i in range(numPeaks):    
        
        if i == highestPeak:
            C[i,:] = 1e12
            C[i,sampleProms >= peakElevs.max()] = 0
            continue
            
        for j in range(numSamples):
            
            # prominence not allowed, continue
            if sampleProms[j] < peakRangeProm[i,0] or sampleProms[j] > peakRangeProm[i,1] or sampleProms[j] > peakElevs[i]:
                C[i,j] = 1e12
                continue
            
            dElev = sampleProms[j] - peakProms[i]
            
            # lowering saddle we can always get any prom
            # or increasing the saddle up to the maximum height
            if dElev >= 0 or np.abs(dElev) < saddleMaxIncrease[peakSaddle[i]]:
                cost   = np.abs(dElev)**2 / (1000*saddleWidth[peakSaddle[i]])
                C[i,j] = cost

            # otherwise, we need to reduce peak height (but we modify elevs pdf)
            else:
                C[i,j] = 1e12
                    

    return C
    

def matchProminences(peakElevs, peakProms, peakRangeProm, sampledProms, peakSaddle, 
                     saddleInitElevs, saddleMaxElevs, saddleWidth):

    numPeaks = peakElevs.size
    saddleMaxIncrease = saddleMaxElevs - saddleInitElevs

    # account for highest peak
    if sampledProms.max() < peakElevs.max():
        sampledProms[sampledProms.argmin()] = peakElevs.max()
    
    # compute cost matrix
    PromCostMatrix = computeCostProminences(peakElevs, peakProms, peakRangeProm, sampledProms,
                                            peakSaddle, saddleMaxIncrease, saddleWidth)
    
    # solver using optimal transport, optimal reached if enough iters
    a = np.ones((numPeaks,))
    b = np.ones((numPeaks,))
    otMatrix = ot.emd(a, b, PromCostMatrix, numItermax=1000000000)
    otDest = otMatrix.argmax(axis=1)
    otCost = PromCostMatrix[np.arange(numPeaks), otDest]
    destProms = sampledProms[otDest]

    # assign elevations
    saddleElevs = saddleInitElevs
    numChanges  = 0
    for i in range(numPeaks):

        promDst = sampledProms[otDest[i]]
        promSrc = peakProms[i]
        diffH   = promSrc - promDst # to increase prom, reduce saddle
        psaddle = peakSaddle[i]        
        
        if np.abs(diffH) < 10:
            continue      
        
        # modify if possible the saddle
        if diffH < saddleMaxIncrease[psaddle]:
            saddleElevs[psaddle] = np.maximum(saddleElevs[psaddle] + diffH, 0)
            numChanges += 1
        # otherwise, reduce peak elevation
        #else:
        #    peakElevs[i] += np.abs(diffH)
        
    return peakElevs, saddleElevs, numChanges



######################
# 4. MULTISTEP UTILS #
######################

def getRidgesDF(peakCoords, saddleCoords, saddlePeaks, imgSize, terrainSize, ridgesWidth=4, normalized=True):
    imgRidges = np.ones(imgSize)
    for i in range(saddleCoords.shape[0]):
        p1 = (peakCoords[saddlePeaks[i,0]]/terrainSize*imgSize).astype(int)
        p2 = (peakCoords[saddlePeaks[i,1]]/terrainSize*imgSize).astype(int)
        ps = (saddleCoords[i]/terrainSize*imgSize).astype(int)
        # OpenCV inverts x and y!
        cv.line(imgRidges, (p1[1], p1[0]), (ps[1], ps[0]), color=0, thickness=ridgesWidth)
        cv.line(imgRidges, (p2[1], p2[0]), (ps[1], ps[0]), color=0, thickness=ridgesWidth)    
    ridgesDF = cv.distanceTransform(imgRidges.astype(np.uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE)
    if normalized:
        cv.normalize(ridgesDF, ridgesDF, 0, 1.0, cv.NORM_MINMAX)
    return ridgesDF, imgRidges
    

def getRiversFromRidges(imgRidges, ridgesWidth=16, riversWidth=4, normalized=True):

    b = 32
    biggerRidges = cv.copyMakeBorder(imgRidges, top=b, bottom=b, left=b, right=b, borderType=cv.BORDER_CONSTANT, value=1)
    kernel = np.ones((5,5),np.uint8)
    biggerRidges = cv.erode(biggerRidges, kernel, iterations=ridgesWidth)

    axesMedial = 1 - medial_axis(biggerRidges > 0, return_distance=False)
    axesMedial = cv.erode(axesMedial.astype(np.uint8), kernel, iterations=riversWidth)
    axesMedial = axesMedial[b:-b, b:-b]
    riversDF   = cv.distanceTransform(axesMedial.astype(np.uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE)
    if normalized:
        cv.normalize(riversDF, riversDF, 0, 1.0, cv.NORM_MINMAX)

    return riversDF
    
    
def smoothstep(x, mi, mx): 
    return (lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )


#############
# SYNTHESIS #
#############


def synthDivideTree(distributions, distribsPromBin, distribsPromAcc, promGroups, stepPeaks,  
                    probMap, probMapSaddles, elevMap, fixedData, synthParams):

    fixedPeaks    = fixedData['fixedPeaks']
    peakRangeProm = fixedData['peakRangeProm']
    peakRangeDom  = fixedData['peakRangeDom']
    fixedSaddles  = fixedData['fixedSaddles']
    fixedSaddlesPeaks = fixedData['fixedSaddlesPeaks']

    promEpsilon = synthParams.get('promEpsilon', 30)
    globalMaxElev = synthParams.get('globalMaxElev', 9000)
    
    elevRangeFilter = synthParams.get('elevRangeFilter', 0.15)
    maxPeakTrials = synthParams.get('maxPeakTrials', 100)
    elevMapDeviation = synthParams.get('elevMapDeviation', 0.05)
    delaunayRidgeExp = synthParams.get('delaunayRidgeExp', 1)
    
    valleyFactor =  synthParams.get('valleyFactor', 1)
    updateProbMap = synthParams.get('updateProbMap', True) and valleyFactor > 0
    
    numPromSteps  = len(promGroups)
    numHistogramIters = synthParams['numHistogramIters']
    terrainSize = synthParams['terrainSize']

    inputProbMap = probMap.copy()
    stepDivtrees = []
    stepProbMaps = []
    
    for gi in range(numPromSteps):
        
        distributionsBin = distribsPromBin[gi]
        distributionsAcc = distribsPromAcc[gi]
        minProm, maxProm = promGroups[gi]
        numPeaks = stepPeaks[gi]
        print('Prom [%d, %d): %d/%d peaks' % (minProm, maxProm, numPeaks, np.sum(stepPeaks)))        


        # 1. place peaks   
        t0 = time.perf_counter()
        peakCoords, peakElevs = generatePeaks(numPeaks, terrainSize, distributionsBin, probMap, elevMap, 
                                              fixedPeaks, elevRangeFilter, elevMapDeviation,
                                              int(maxPeakTrials*(1 - 0.5*(gi/np.maximum(1, numPromSteps - 1)))))
        # add fixed peaks
        peakCoords = np.concatenate([fixedPeaks[:,:2], peakCoords])
        peakElevs  = np.concatenate([fixedPeaks[:,2].squeeze(), peakElevs])
        print('Peaks', '%.2f s'%(time.perf_counter() - t0))


        # 2. MST
        t0 = time.perf_counter()  
        saddleCoords, saddlePeaks, saddleWeights = createDelaunayGraph(peakCoords, peakElevs, probMapSaddles, 
                                                                       terrainSize, distributionsAcc, delaunayRidgeExp)
                                                                       
        # add for fixed ridges with 0 cost
        saddleCoords = np.vstack([saddleCoords, fixedSaddles[:,:2]])
        saddlePeaks  = np.vstack([saddlePeaks, fixedSaddlesPeaks])
        saddleWeights = np.vstack([saddleWeights, np.tile([1e-12], (fixedSaddles.shape[0],1))])

        # keep the MST for a ridgeline tree (divide tree)
        RidgeTree, saddleCoords, saddlePeaks, saddleMaxElev, saddleWidth = generateRidgeTree(peakCoords, peakElevs, 
                                                                                             saddleCoords, saddlePeaks, 
                                                                                             saddleWeights)
        numSaddles  = saddleMaxElev.size
        saddleElevs = saddleMaxElev    
        print('Ridges', '%.2f s'%(time.perf_counter() - t0))


        # information about this step ranges
        #peakRangeProm = np.vstack([peakRangeProm, np.tile([minProm, maxProm], (numPeaks, 1))])
        peakRangeProm = np.vstack([peakRangeProm, np.tile([0, globalMaxElev], (numPeaks, 1))])
        peakRangeDom  = np.clip(peakRangeProm/peakElevs[:,np.newaxis], 0, 1)
        
        stepDivtrees.append((peakCoords.copy(), peakElevs.copy(), saddleCoords.copy(), saddlePeaks.copy()))        
        stepProbMaps.append(probMap.copy())
        
        if gi < numPromSteps-1:

            t0 = time.perf_counter()  

            # Pass information to next iteration      
            # use generated peaks as fixed peaks on next step
            fixedPeaks = np.hstack([peakCoords, peakElevs[:,np.newaxis]])
            # we do not pass the saddles, as new peaks inbetween might appear and disconnect them
            #fixedSaddles = np.hstack([saddleCoords, saddleElevs[:,np.newaxis]])

            # interpolation factor
            ti = gi/(numPromSteps-1)#fixedPeaks.shape[0]/totalNumPeaks

            # use current ridgelines to guide probabilities
            dfShape = probMap.shape
            dfPixSize = np.max(1000*terrainSize/dfShape)
            ridgesWidth = (1/valleyFactor)*((3000 - 2000*ti)/dfPixSize).astype(int)  # (3000 to 1000 m) / valleyFactor
            riversWidth = valleyFactor*((1000 - 900*ti)/dfPixSize).astype(int)       # (1000 to  200 m) * valleyFactor
            
            dfRidges, imgRidges = getRidgesDF(peakCoords, saddleCoords, saddlePeaks, dfShape, terrainSize, ridgesWidth=1, normalized=False)        
            dfRivers = getRiversFromRidges(imgRidges, ridgesWidth=1, riversWidth=1, normalized=False)

            if updateProbMap:
                # make next iteration peaks more probable close to current ridges
                probRidge = 1 - smoothstep(dfRidges, 0.25*ridgesWidth, 4*ridgesWidth)
                #probMap = probMap * probRidge

                # mask rivers from probability map so we do not place peaks in the valleys
                probRiver = smoothstep(dfRivers, 0.25*riversWidth, 4*riversWidth)
                probMap = inputProbMap * probRiver

            print('Info propagation step', '%.2f s'%(time.perf_counter() - t0))

        else:

            # 3: Adjust saddle elevations to match prom/dom      

            # compute initial prominence
            t0 = time.perf_counter() 
            saddleRndOff   = np.random.rand(numSaddles,1).squeeze()  # for unique key saddle computation
            saddleMaxElev -= 1
            saddleElevs    = saddleMaxElev - saddleRndOff - promEpsilon
            peakSaddle, peakParent, peakProms, KSMatrix = computeProminences(RidgeTree, peakElevs, saddleElevs, saddlePeaks)
            peakDoms = peakProms / peakElevs
            print('Prominences', '%.2f s'%(time.perf_counter() - t0))

            # aim for target dominances
            sampledDoms  = sampleFromPDF(peakDoms.size, distributions['dominance']['hist'],
                                         distributions['dominance']['bins'])
            sampledProms = sampleFromPDF(peakProms.size, distributions['prominence']['hist'],
                                         distributions['prominence']['bins'])

            histIter = 0

            domDifferences  = []
            promDifferences = []
            prevSaddleElevs = []
            domObjectiveHist  = np.round(peakDoms.size * (distributions['dominance']['hist']
                                                         /distributions['dominance']['hist'].sum()))
            promObjectiveHist = np.round(peakProms.size * (distributions['prominence']['hist']
                                                          /distributions['prominence']['hist'].sum()))              
            domDifferences.append(np.sum(np.abs(np.histogram(peakDoms,   bins=distributions['dominance']['bins'])[0]
                                              - domObjectiveHist)))
            promDifferences.append(np.sum(np.abs(np.histogram(peakProms, bins=distributions['prominence']['bins'])[0]
                                              - promObjectiveHist)))
            prevSaddleElevs.append(saddleElevs.copy())
            
            print('* Initial errors --> %d, %d' % (domDifferences[-1], promDifferences[-1]))

            while histIter <= numHistogramIters:

                if histIter%2 == 0:
                    iterType = 'P'
                else:
                    iterType = 'D'

                t0 = time.perf_counter() 
                if iterType == 'D':
                    # match dominances
                    peakElevs, saddleElevs, changes = matchDominances(peakElevs, peakDoms, peakRangeDom, sampledDoms, 
                                                                      peakSaddle, saddleElevs, saddleMaxElev, saddleWidth)
                elif iterType == 'P':
                    # match prominences
                    peakElevs, saddleElevs, changes = matchProminences(peakElevs, peakProms, peakRangeProm, sampledProms, 
                                                                       peakSaddle, saddleElevs, saddleMaxElev, saddleWidth)
                print('Match prom/dom', '%.2f s'%(time.perf_counter() - t0))

                # correct numeric errors
                saddleElevs = np.minimum(saddleElevs, saddleMaxElev)

                # fixed saddles
                for si,speaks in enumerate(fixedSaddlesPeaks):
                    s = RidgeTree[speaks[0], speaks[1]]
                    saddleElevs[s] = fixedSaddles[si,2]

                # update values
                t0 = time.perf_counter() 
                peakSaddle, peakParent, peakProms, KSMatrix = computeProminences(RidgeTree, peakElevs, saddleElevs, saddlePeaks)
                peakDoms = peakProms / peakElevs
                print('Prominences', '%.2f s'%(time.perf_counter() - t0))

                #print('Free peaks', np.sum(np.sum(KSMatrix >= 0, axis=1) == 1))

                domDifferences.append(np.sum(np.abs(np.histogram(peakDoms,   bins=distributions['dominance']['bins'])[0]
                                                  - domObjectiveHist)))
                promDifferences.append(np.sum(np.abs(np.histogram(peakProms, bins=distributions['prominence']['bins'])[0]
                                                  - promObjectiveHist)))
                prevSaddleElevs.append(saddleElevs.copy())

                histIter += 1

                print('* Step %d (%s): changed %d --> %dd + %dp = %d' % (histIter, iterType, changes, 
                                                                         domDifferences[-1],  promDifferences[-1], 
                                                                         domDifferences[-1] + promDifferences[-1]))

            # keep the best saddles configuration
            bestSaddles = (np.array(promDifferences) + np.array(domDifferences)).argmin()        
            print('Keeping saddles %d: %d errors' % (bestSaddles, promDifferences[bestSaddles] + domDifferences[bestSaddles]))

            saddleElevs = prevSaddleElevs[bestSaddles]
            peakSaddle, peakParent, peakProms, _ = computeProminences(RidgeTree, peakElevs, saddleElevs, saddlePeaks)   

            # ensure minimum prominence
            promAdjustment = np.maximum(0, promEpsilon - peakProms[peakSaddle >= 0])
            saddleElevs[peakSaddle[peakSaddle >= 0]] -= promAdjustment
            peakProms[peakSaddle >= 0] += promAdjustment


    debugInfo = {
        'stepDivtrees': stepDivtrees,
        'stepProbMaps': stepProbMaps,
        'domDifferences': domDifferences,
        'promDifferences': promDifferences
    }
            
    return peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks, RidgeTree, debugInfo
    