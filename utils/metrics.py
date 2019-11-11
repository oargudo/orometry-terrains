import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from queue import Queue


def computeProminences(RidgeTree, peakElevs, saddleElevs, saddlePeaks):

    numPeaks = peakElevs.size
    peaksByHeight = np.argsort(peakElevs)[::-1]
    
    # RidgeTree[i,j] == saddle id connecting peak i to peak j, -1 ortherwise
    # convert to csr representation with 0 on no-links
    TreeSaddles = RidgeTree + 1
    TreeSaddles[np.triu_indices(numPeaks, 1)] = 0
    TreeSaddles = csr_matrix(TreeSaddles)
    
    # This matrix stores the peak-saddle relation
    KeySaddleMatrix = np.full((numPeaks, numPeaks), -1)
    KeySaddleMatrix[peaksByHeight[0], numPeaks-1] = peakElevs[peaksByHeight[0]]

    # prominence for each peak
    promSaddle = np.full((numPeaks, ), -1)
    promParent = np.full((numPeaks, ), -1)
    promValues = np.full((numPeaks, ), -1.0)
    
    promValues[peaksByHeight[0]] = peakElevs[peaksByHeight[0]]

    for i in range(1, numPeaks):

        # peak id
        pi = int(peaksByHeight[i])

        # all paths from pi to all other peaks
        _, preds = dijkstra(TreeSaddles, indices=pi, directed=False, unweighted=True, return_predecessors=True)
        
        # BFS from pi until a higher peak is reached
        higherPeaks = []
        neighSaddles = RidgeTree[pi, RidgeTree[pi,:] >= 0]
        Q = Queue()
        for n in neighSaddles:
            Q.put((n, pi))        
        while (Q.qsize() > 0):        
            # get saddle and the two peaks 
            # s.t. p0 is where we come from, and p1 where we go
            s,p0 = Q.get()  
            p1 = saddlePeaks[s,0] if saddlePeaks[s,0] != p0 else saddlePeaks[s,1]
            if peakElevs[p1] < peakElevs[pi]:
                # if lower, keep exploring
                neighSaddles = RidgeTree[p1, RidgeTree[p1,:] >= 0] 
                for n in neighSaddles:
                    if n != s:
                        Q.put((n, p1))
            else:
                # if higher, stop search here
                higherPeaks.append(p1)

        # find prominence 
        pKeySaddle = -1
        pPromParent = -1
        hKeySaddle = 0  
        
        # for all reachable higher peaks
        for pj in higherPeaks:
        
        #for j in range(i):
            #pj = int(peaksByHeight[j])

            # reconstruct path
            path = []
            curr = pj
            while curr != pi:
                path.append(curr)
                curr = preds[curr]
            path.append(pi)
            path = path[::-1]

            # follow path pi->pj stopping at first higher peak, and keep lowest saddle
            minSaddle = -1
            minSaddleElev = peakElevs[pi] + 0.1
            pathParent = -1
            for k in range(1, len(path)):
                s = RidgeTree[path[k-1], path[k]]
                if minSaddleElev > saddleElevs[s]:
                    minSaddleElev = saddleElevs[s]
                    minSaddle = s
                if peakElevs[path[k]] > peakElevs[pi]:
                    pathParent = path[k]
                    break
                    
            # check if this saddle elevation is higher than previously found
            if minSaddleElev > hKeySaddle:
                pKeySaddle = minSaddle
                hKeySaddle = minSaddleElev
                pPromParent = pathParent
                
            # annotate all saddles along the path with the prominence
            for k in range(1, len(path)):
                s = RidgeTree[path[k-1], path[k]]
                KeySaddleMatrix[pi, s] = peakElevs[pi] - minSaddleElev
                if peakElevs[path[k]] > peakElevs[pi]:
                    break
                
            if minSaddle < 0:
                print('bad saddle', i, pi, pj, path)


        promSaddle[pi] = pKeySaddle
        promParent[pi] = pPromParent
        promValues[pi] = peakElevs[pi] - saddleElevs[pKeySaddle]
    
    return promSaddle, promParent, promValues, KeySaddleMatrix


def computeIsolations(peakCoords, peakElevs):
 
    numPeaks = peakElevs.size
    peaksByHeight = np.argsort(peakElevs)[::-1]
    
    isolDists  = np.full((numPeaks, ), -1.0)
    isolCoords = np.full((numPeaks, 2),  0.0)
    
    for i in range(1, numPeaks):

        # peak id
        pi = int(peaksByHeight[i])
        ci = peakCoords[pi]

        # find isolation
        isolation = 1e6
        cIsol = [0, 0]
        
        # for all higher peaks
        for j in range(i):

            pj = int(peaksByHeight[j])

            # compare distance and update isolation
            cj = peakCoords[pj]
            d = np.linalg.norm(ci - cj)
            if d < isolation:
                isolation = d
                cIsol = cj
    
        # store result
        isolDists[pi]  = isolation
        isolCoords[pi] = cIsol
        
    return isolDists, isolCoords
