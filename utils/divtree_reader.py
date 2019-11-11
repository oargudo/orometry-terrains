import numpy as np
from utils.coords import *


def readDivideTree(filepath, crop=[], returnDEMCoords=False):
    
    fin = open(filepath, 'r')
    
    doCrop = False
    if len(crop) > 0:
        doCrop = True
        latMin, lonMin, latMax, lonMax = crop

    numPeaks = int(fin.readline().split()[1])
    peakCoords = np.zeros((numPeaks,2))
    peakElevs = np.zeros((numPeaks,))    
    idPeak = 0
    peakReorder = np.full((numPeaks,), -1)
    for i in range(numPeaks):
        peak = fin.readline().split()
        peakCoords[i, 0] = float(peak[1])
        peakCoords[i, 1] = float(peak[2])
        peakElevs [i]    = feet2m(float(peak[3]))
        if doCrop:
            if peakCoords[i,0] < lonMin or peakCoords[i,1] < latMin or peakCoords[i,0] > lonMax or peakCoords[i,1] > latMax:
                continue
        peakReorder[i] = idPeak
        
        if returnDEMCoords:
            peakCoords[i, 0] = float(peak[4])
            peakCoords[i, 1] = float(peak[5])
        
        idPeak += 1

    numSaddles = int(fin.readline().split()[1])
    saddleCoords = np.zeros((numSaddles,2))
    saddleElevs  = np.zeros((numSaddles,))
    for i in range(numSaddles):
        saddle = fin.readline().split()
        saddleCoords[i, 0] = float(saddle[1])
        saddleCoords[i, 1] = float(saddle[2])
        saddleElevs [i]    = feet2m(float(saddle[3]))
        if returnDEMCoords:
            saddleCoords[i, 0] = float(saddle[4])
            saddleCoords[i, 1] = float(saddle[5])

    numBasinSaddles = int(fin.readline().split()[1])
    for i in range(numBasinSaddles):
        saddle = fin.readline().split()

    numRunoff = int(fin.readline().split()[1])
    runoffs = []
    for i in range(numRunoff):
        saddle = fin.readline().split()

    RidgeTree = np.full((numPeaks, numPeaks), -1)
    saddlePeaks = np.full((numSaddles, 2), -1)
    numEdges = int(fin.readline().split()[1])
    for i in range(numEdges):
        edge = fin.readline().split()
        epeak   = peakReorder[int(edge[0])-1]
        eparent = peakReorder[int(edge[1])-1]
        esaddle = int(edge[2])-1
        if epeak < 0 or eparent < 0:
            continue
        if epeak < numPeaks and eparent < numPeaks and esaddle < numSaddles:
            saddlePeaks[esaddle,:] = [epeak, eparent]
            RidgeTree[epeak, eparent] = RidgeTree[eparent, epeak] = esaddle
        else:
            print('Bad edge', edge)

    fin.close()
    
    
    peakCoords = peakCoords[peakReorder >= 0,:]
    peakElevs  = peakElevs[peakReorder >= 0]
    numPeaks   = peakElevs.size
    
    saddleReorder = np.full((numSaddles,), -1)    
    idSaddle = 0
    for i in range(numSaddles):
        if numPeaks > saddlePeaks[i,0] >= 0 and numPeaks > saddlePeaks[i,1] >= 0:
            RidgeTree[RidgeTree == i] = idSaddle
            saddleReorder[i] = idSaddle
            idSaddle += 1
    saddleCoords = saddleCoords[saddleReorder >= 0,:]
    saddleElevs  = saddleElevs[saddleReorder >= 0]
    saddlePeaks  = saddlePeaks[saddleReorder >= 0, :]        

    
    return peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks, RidgeTree
