import numpy as np
import triangle
import time
import shapely
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import linemerge, unary_union, polygonize
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, cKDTree
from scipy.spatial.distance import cdist
from queue import Queue


###########
# VORONOI #
###########

# adapted from: https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    ridge_regions = {}
    ridge_new_vert = {}

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
        
        ridge_regions[(v1,v2)] = (p1,p2)
        ridge_regions[(v2,v1)] = (p1,p2)

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue
                
            # check if we already created this vertex from the opposite direction
            newVertId = ridge_new_vert.get((p1,p2), -1)
            if newVertId >= 0:
                new_region.append(newVertId)
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            newVertId = len(new_vertices)
            new_region.append(newVertId)
            new_vertices.append(far_point.tolist())
            ridge_regions[(v2, newVertId)] = (p1,p2)
            ridge_regions[(newVertId, v2)] = (p1,p2)
            ridge_new_vert[(p1,p2)] = newVertId
            ridge_new_vert[(p2,p1)] = newVertId
            
            
        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices), ridge_regions


def getClippedVoronoiCells(vor, bbox):
    
    vorclipRegions, vorclipVertices, vorEdgesRegions = voronoi_finite_polygons_2d(vor)

    polygons = []
    for r in vorclipRegions:
        rverts = vorclipVertices[r]
        poly = Polygon(rverts)
        poly = poly.intersection(bbox)
        polygons.append(poly)  
    
    return polygons, vorclipRegions, vorclipVertices, vorEdgesRegions


def getVirtualRidgePoints(pA, pB, maxLength, randomOff):
    
    rlen = np.linalg.norm(pB - pA)
    numParts   = int(rlen/maxLength)
    partLength = rlen/(numParts + 1)
    segDir     = (pB - pA)/rlen
    
    points = []
    for i in range(numParts):
        r = np.random.uniform(-randomOff, randomOff)
        p = pA + partLength*(i + 1 + r)*segDir
        points.append([p[0], p[1]])
    
    return points


############################
# RIVER AND RIDGE NETWORKS #
############################

def buildVoronoiRivers(clippedVertices, clippedRegions, voronoiRidgeRegions,
                       peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks,
                       terrainBbox, terrainSize):

    # maximum river elevation init value, any number > max(peakElevs)
    maxGlobalElev  = peakElevs.max()*10
    minTerrainElev = saddleElevs.min()
    numPeaks = peakElevs.size
    numSaddles = saddleElevs.size

    # adjacency matrix of the river network
    numVertices  = len(clippedVertices)
    RiverVertAdj = np.full((numVertices, numVertices), False)
    riverMaxElev = np.full((numVertices, ), maxGlobalElev)
    riverFlowTo  = np.full((numVertices, ), -1)
    riverDrainArea = np.zeros((numVertices,))
    
    
    # start with all connected voronoi segments, and compute vertex drainage area
    for region in clippedRegions:
        
        try:
            regionArea = Polygon(np.clip(clippedVertices[region], [0,0], terrainSize)).area
        except:
            regionArea = 0
        
        for i in range(len(region)):    
            # river connectivity
            v1  = region[i]
            v2  = region[(i+1)%len(region)]
            RiverVertAdj[v1,v2] = RiverVertAdj[v2,v1] = True
            
            # approximate drainage area on vertices
            # note we will do these twice per segment because area is different per face
            # the area is not really distributed equivalently per vertex on the cell
            riverDrainArea[v1] += regionArea/len(region)
            
                    
    # visit all ridges from divide tree, intersect regions, and remove connectivity
    ridges = []
    for si,[p1,p2] in enumerate(saddlePeaks):
        ridges.append((p1,si))
        ridges.append((p2,si))
        
    # ridge to voronoi intersections
    for p,s in ridges:
        
        line = LineString([peakCoords[p], saddleCoords[s]])
        
        # indices of starting and ending regions of this segment
        regionIni = p
        regionEnd = s + numPeaks
        
        rprev = -1
        ri = regionIni
        while ri != regionEnd:
            
            # check all region polygon edges
            region = clippedRegions[ri]
            for i in range(len(region)):
                v1  = region[i]
                v2  = region[(i+1)%len(region)]
                seg = LineString([clippedVertices[v1], clippedVertices[v2]])

                # intersection or voronoi point on a ridge (very rare, but happened!)
                if seg.intersects(line) or seg.distance(line) < 1e-6:

                    # remove river adjacency
                    RiverVertAdj[v1,v2] = RiverVertAdj[v2,v1] = False

                    # lower maximum allowed elevation of voronoi vertices if needed
                    riverMaxElev[v1] = np.minimum(riverMaxElev[v1], saddleElevs[s])
                    riverMaxElev[v2] = np.minimum(riverMaxElev[v2], saddleElevs[s])

                    # if this is not where we came from, go to process next region
                    segRegions = voronoiRidgeRegions[(v1,v2)]
                    rnext = segRegions[0] if ri == segRegions[1] else segRegions[1]
                    if rnext != rprev:
                        rprev = ri
                        ri = rnext
                        break
    
    
    # transform river vertex adjacency into a DAG defining the river flow
    RiverDAG = RiverVertAdj.copy()

    # the river sources are the vertices that have an elevation defined by
    # a nearby ridge, as well as being a topological leaf of the graph
    riverSources = np.logical_and(riverMaxElev < maxGlobalElev, np.sum(RiverVertAdj, axis=0) == 1).nonzero()[0]
    
    # init queue with the river sources
    Q = Queue()
    riverEndpoints = []
    for rs in riverSources:
        if terrainBbox.contains(Point(clippedVertices[rs])):
            Q.put((rs, maxGlobalElev))
        else:
            riverEndpoints.append(rs)
    riverSources = [rs for rs in riverSources if not rs in riverEndpoints]
    
    # eliminate incoming edges on river sources
    for rs in riverSources:
        RiverDAG[:,rs] = False
    
    # explore the river network to extract parents
    while Q.qsize() > 0:

        # get vertex, and propagated elevation from child
        nodeId,riverElev = Q.get()

        # update max possible elevation for this node
        riverMaxElev[nodeId] = np.minimum(riverMaxElev[nodeId], riverElev)

        # possible destinations
        dests = RiverDAG[nodeId,:].nonzero()[0]
        if len(dests) == 1:
            # if only one possible destination, there we move
            parent = dests[0]  
            riverFlowTo[nodeId] = parent   
            
            # eliminate backward edge from parent
            RiverDAG[parent, nodeId] = False

            # add to queue
            Q.put((parent, riverMaxElev[nodeId]))
            
            
    # propagate drainage area
    rparents = np.zeros((numVertices,))
    for rto in riverFlowTo:
        rparents[rto] += 1    
    Q = Queue()
    for i in range(rparents.size):
        if rparents[i] == 0:
            Q.put(i)
    while Q.qsize() > 0:
        rfrom = Q.get()
        rto   = riverFlowTo[rfrom]
        riverDrainArea[rto] += riverDrainArea[rfrom]
        rparents[rto] -= 1
        if rparents[rto] == 0:
            Q.put(rto)

    return riverMaxElev, riverFlowTo, riverSources, riverEndpoints, riverDrainArea, RiverVertAdj


def getRiverSlopeElev(elevation, maxElevation, minElevation):
    normElev = np.maximum(0, (elevation - minElevation)/(maxElevation - minElevation))
    return np.maximum(np.random.normal(normElev, 0.1*normElev), 0)


def getRiverSlopeDrain(drainage, maxDrainage):
    normDrain = (drainage/maxDrainage)
    return (1 - normDrain)


def getRiverSlopeDistance(distTravel):
    normDist = np.minimum(distTravel/30.0, 1.0)
    return (1 - normDist)


def getRiverHeights(riverMaxElev, riverSources, riverFlowTo, riverDrainArea, clippedVertices, bbox, 
                    slopeCoeff=0.3, minRiverElev=0, srcOffsetMean=50, srcOffsetStd=20):
    # assign valley elevations following the rivers
    riverElevs = riverMaxElev.copy()

    # river sources: assign random elevation
    for i,rs in enumerate(riverSources):
        # assign elev
        riverElevs[rs] = np.maximum(minRiverElev, 
                                    riverMaxElev[rs] - np.random.normal(srcOffsetMean,srcOffsetStd))


    # travel along the flow
    nodeQueued = np.full(riverElevs.shape, False)
    Q = Queue()
    for rs in riverSources:
        Q.put((rs, riverFlowTo[rs], 0))
        nodeQueued[rs] = True

    while Q.qsize() > 0:

        # get source and dest nodes
        nodeSrc, nodeDst, sourceDist = Q.get()
        if nodeDst < 0:
            continue
        psrc = clippedVertices[nodeSrc,:]
        pdst = clippedVertices[nodeDst,:]

        # compute travelled distance and slope 
        if bbox.contains(Point(pdst)) and bbox.contains(Point(psrc)):
            dist = np.linalg.norm(psrc - pdst)
        else:
            dist = 0
        slopeE = getRiverSlopeElev(np.minimum(riverMaxElev[nodeDst], riverElevs[nodeSrc]), riverMaxElev.max(), minRiverElev)
        slopeA = getRiverSlopeDrain(riverDrainArea[nodeSrc], riverDrainArea.max())
        slopeD = getRiverSlopeDistance(sourceDist)
        slope  = slopeCoeff*(0.5*slopeE + 0.5*slopeA)
        
        dstElev = riverElevs[nodeSrc] - 1000*dist*slope 
        dstElev = np.maximum(minRiverElev, dstElev)
        
        # if we update elevation, re-flow
        loweredElev = dstElev < riverElevs[nodeDst]
        riverElevs[nodeDst] = np.minimum(dstElev, riverElevs[nodeDst])
        
        # propagate if not visited or if lowered the elevation of dst
        if not nodeQueued[nodeDst] or loweredElev:
            Q.put((nodeDst, riverFlowTo[nodeDst], sourceDist + dist))
            nodeQueued[nodeDst] = True
    
    return riverElevs


def propagateRiverFlowElev(riverSources, riverFlowTo, riverElevs):
    
    # travel along the flow
    Q = Queue()
    for rs in riverSources:
        Q.put((rs, riverFlowTo[rs]))

    while Q.qsize() > 0:

        # get source and dest nodes
        nodeSrc, nodeDst = Q.get()

        # fix elevation
        riverElevs[nodeDst] = np.minimum(riverElevs[nodeSrc], riverElevs[nodeDst])

        # keep propagating
        if riverFlowTo[nodeDst] >= 0:
            Q.put((nodeDst, riverFlowTo[nodeDst]))
            
    return riverElevs


def smoothRiverElevs(riverInitElevs, riverMaxElevs, RiverVertAdj, smoothIters=10, carveOnly=False, sourcesMomentum=0.0):
    
    smoothElevs = riverInitElevs.copy()
    
    vertNeighs = np.sum(RiverVertAdj, axis=0)
    for _ in range(smoothIters): 
        laplElevs = np.dot(RiverVertAdj, smoothElevs)
        laplElevs[vertNeighs > 1] = laplElevs[vertNeighs > 1]/vertNeighs[vertNeighs > 1]
        smoothElevs[vertNeighs == 1] =      sourcesMomentum * smoothElevs[vertNeighs == 1] + \
                                       (1 - sourcesMomentum)* laplElevs[vertNeighs == 1] 
        if carveOnly:
            # we never allow increasing the elevation
            smoothElevs[vertNeighs > 1] = np.minimum(smoothElevs[vertNeighs > 1], laplElevs[vertNeighs > 1])
        else:
            # we allow increasing the elev, as long as the maximum is not surpassed
            smoothElevs[vertNeighs > 1] = np.minimum(riverMaxElevs[vertNeighs > 1], laplElevs[vertNeighs > 1])        
                
    return smoothElevs


def smoothRiverPositions(riverCoords, RiverAdj, smoothIters=10, sourcesMomentum=1.0):

    smoothCoords = riverCoords.copy()
    vertNeighs = np.sum(RiverAdj, axis=0)
    
    for _ in range(smoothIters): 
        laplCoords = np.dot(RiverAdj, smoothCoords)
        smoothCoords[vertNeighs > 1,:] = laplCoords[vertNeighs > 1,:]/vertNeighs[vertNeighs > 1, np.newaxis]
        smoothCoords[vertNeighs == 1,:] = sourcesMomentum*smoothCoords[vertNeighs == 1,:] + \
                                          (1 - sourcesMomentum)*laplCoords[vertNeighs == 1,:] 
                
    return smoothCoords


def refineRiverNetwork(riverFlowTo, netVertices, riverMaxElev, riverDrainArea, riverSources, splitLength, perturbScale, bbox):
    
    # refined river network
    riverCoords = np.empty((0,2))
    riverElevs  = np.empty((0,))
    riverDrainA = np.empty((0,))
    riverSegs   = []
    riverVertId = np.full(riverFlowTo.shape, -1)

    numRiverVerts = 0
    for rfrom,rto in enumerate(riverFlowTo):
        if rto >= 0:
            p1 = netVertices[rfrom,:]
            p2 = netVertices[rto,:]
            segLength = np.linalg.norm(p1 - p2)
            segDir = (p2 - p1)/segLength
            ridgeIndices = []

            # startpoint
            if riverVertId[rfrom] < 0:
                riverCoords = np.vstack([riverCoords, p1])
                riverElevs  = np.hstack([riverElevs, riverMaxElev[rfrom]])
                riverDrainA = np.hstack([riverDrainA, riverDrainArea[rfrom]])
                riverVertId[rfrom] = numRiverVerts
                rCurr = numRiverVerts
                numRiverVerts += 1
            else:
                rCurr = riverVertId[rfrom]
            ridgeIndices.append(rCurr)


            # subdivide river if necessary
            t = splitLength + 0.5*np.random.uniform(-1, 1)*splitLength
            while t < segLength:

                # interpolation
                p = p1 + t*segDir
                e = (1 - t/segLength)*riverMaxElev[rfrom] + (t/segLength)*riverMaxElev[rto]
                a = (1 - t/segLength)*riverDrainArea[rfrom] + (t/segLength)*riverDrainArea[rto]

                # random side deviation
                p += 0.5*np.random.uniform(-1, 1)*splitLength*np.array([segDir[1], -segDir[0]])

                # omit unnecessary vertices
                #if 0 <= p[0] < terrainSize[0] and 0 <= p[1] < terrainSize[1]:
                if bbox.contains(Point(p)):
                    riverCoords = np.vstack([riverCoords, p])
                    riverElevs  = np.hstack([riverElevs, e])
                    riverDrainA = np.hstack([riverDrainA, a])
                    riverSegs.append([rCurr, numRiverVerts])
                    rCurr = numRiverVerts
                    ridgeIndices.append(rCurr)
                    numRiverVerts += 1

                t += splitLength + 0.5*np.random.uniform(-1, 1)*splitLength

            # endpoint
            if riverVertId[rto] < 0:
                riverCoords = np.vstack([riverCoords, p2])
                riverElevs  = np.hstack([riverElevs, riverMaxElev[rto]])
                riverDrainA = np.hstack([riverDrainA, riverDrainArea[rto]])
                riverVertId[rto] = numRiverVerts
                riverSegs.append([rCurr, numRiverVerts])
                ridgeIndices.append(numRiverVerts)
                numRiverVerts += 1
            else:
                riverSegs.append([rCurr, riverVertId[rto]])
                ridgeIndices.append(riverVertId[rto])
            
            # perturb positions
            segmentPerturbation(riverCoords, ridgeIndices, perturbScale)

                
    # compute flow directions
    riverFlows = np.full((numRiverVerts,), -1).astype(np.int32)
    for ffrom, fto in riverSegs:
        riverFlows[ffrom] = fto
    sourcesRemap = riverVertId[riverSources]
    sourcesRemap = sourcesRemap[sourcesRemap >= 0]

    return riverCoords, riverElevs, riverSegs, riverFlows, riverDrainA, sourcesRemap


def refineRidgeNetwork(ridges, vertCoords, vertElevs, splitLength, perturbScale, bbox):
    
    # refined ridge network
    ridgeCoords = np.empty((0,2))
    ridgeElevs  = np.empty((0,))
    ridgeSegs   = []
    ridgeVertId = np.full(vertElevs.shape, -1)

    numRidgeVerts = 0
    for r in ridges:
        rfrom, rto = r
        p1 = vertCoords[rfrom,:]
        p2 = vertCoords[rto,:]
        segLength = np.linalg.norm(p1 - p2)
        segDir = (p2 - p1)/segLength
        ridgeIndices = []

        # startpoint
        if ridgeVertId[rfrom] < 0:
            ridgeCoords = np.vstack([ridgeCoords, p1])
            ridgeElevs  = np.hstack([ridgeElevs, vertElevs[rfrom]])
            ridgeVertId[rfrom] = numRidgeVerts
            rCurr = numRidgeVerts
            numRidgeVerts += 1
        else:
            rCurr = ridgeVertId[rfrom]
        ridgeIndices.append(rCurr)

        # subdivide ridge if necessary
        t = splitLength + np.random.uniform(-1, 1)*splitLength*0.3
        while t < segLength:

            # interpolation
            p = p1 + t*segDir
            e = (1 - t/segLength)*vertElevs[rfrom] + (t/segLength)*vertElevs[rto]

            # omit unnecessary vertices
            if bbox.contains(Point(p)):
                ridgeCoords = np.vstack([ridgeCoords, p])
                ridgeElevs  = np.hstack([ridgeElevs, e])
                ridgeSegs.append([rCurr, numRidgeVerts])
                rCurr = numRidgeVerts
                ridgeIndices.append(rCurr)
                numRidgeVerts += 1

            t += splitLength + np.random.uniform(-1, 1)*splitLength*0.3


        # endpoint
        if ridgeVertId[rto] < 0:
            ridgeCoords = np.vstack([ridgeCoords, p2])
            ridgeElevs  = np.hstack([ridgeElevs, vertElevs[rto]])
            ridgeVertId[rto] = numRidgeVerts
            ridgeSegs.append([rCurr, numRidgeVerts])
            ridgeIndices.append(numRidgeVerts)
            numRidgeVerts += 1
        else:
            ridgeSegs.append([rCurr, ridgeVertId[rto]])
            ridgeIndices.append(ridgeVertId[rto])
            
            
        # perturb positions
        segmentPerturbation(ridgeCoords, ridgeIndices, perturbScale)

    return ridgeCoords, ridgeElevs, ridgeSegs


def segmentPerturbation(coords, indices, scale):
    if len(indices) <= 2:
        return
        
    p1 = coords[indices[0],:]
    p2 = coords[indices[-1],:]
    segLength = np.linalg.norm(p1 - p2)
    segDir    = (p2 - p1)/segLength
    magnitude = scale*segLength
    direction = np.array([-segDir[1], segDir[0]])
    
    imid = int(len(indices)/2)
    pmid = coords[indices[imid],:] + direction*np.random.uniform(-magnitude, magnitude)
    
    for i in range(1,imid):
        t = i/imid
        coords[indices[i]] = t*pmid + (1-t)*p1
    for i in range(imid+1, len(indices)-1):
        t = (i-imid)/(len(indices)-imid)
        coords[indices[i]] = t*p2 + (1-t)*pmid
    coords[indices[imid],:] = pmid
    
    segmentPerturbation(coords, indices[:imid+1], scale)
    segmentPerturbation(coords, indices[imid:], scale)



#################
# MAIN FUNCTION #
#################


def divideTreeToMesh(peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks, terrainSize, poissonSamples, reconsParams):
    
    # constants and input parameters
    numPeaks = peakElevs.size
    numSaddles = saddleElevs.size
    
    minTerrainElev        = reconsParams.get('minTerrainElev', 0)
    maxSlopeCoeff         = reconsParams.get('maxSlopeCoeff', 0.2)
    refineDistance        = reconsParams.get('refineDistance', 0.12)
    riversPerturbation    = reconsParams.get('riversPerturbation', 0.2)
    ridgesPerturbation    = reconsParams.get('ridgesPerturbation', 0.15)
    useDrainageForValleys = reconsParams.get('useDrainageForValleys', True)
    maxRiverWidth         = reconsParams.get('maxRiverWidth', 0.3)
    
    coarseRiverSmoothIters     = reconsParams.get('coarseRiverSmoothIters', 4)
    refinedRiverSmoothIters    = reconsParams.get('refinedRiverSmoothIters', 30)
    refinedRiverSmoothPosIters = reconsParams.get('refinedRiverSmoothPosIters', 1)
    
    srcElevRndMean = reconsParams.get('srcElevRndMean', 50)
    srcElevRndStd  = reconsParams.get('srcElevRndStd', 20)
    srcElevMomentCoarse = reconsParams.get('momentumCoarseRiverSourceElevs', 0.5)
    srcElevMoment       = reconsParams.get('momentumRiverSourceElev', 0.75)
    srcCoordsMoment     = reconsParams.get('momentumRiverSourceCoords', 0.7)
    
    virtualRidgePointsDist = reconsParams.get('virtualRidgePointsDist', None)
    
    
    print('Reconstructing terrain with %d peaks'%numPeaks)
    
    # output debug data
    debugInfo = {
        'timings': []
    }
    print(('STAGE NAME', 'run time (s)'))
    
    
    # terrain bounding box extending slithgly the size of the terrain
    boxOff = 0.01*np.array(terrainSize)
    bbox = Polygon([[-boxOff[0], -boxOff[1]], [-boxOff[0], terrainSize[1] + boxOff[1]], 
                   [terrainSize[0] + boxOff[0], terrainSize[1] + boxOff[1]], [terrainSize[0] + boxOff[0], -boxOff[1]]])
        

    # 1. Compute Voronoi cells of the peak and saddles graph
    
    # subdivide ridges with "virtual" points for Voronoi. 
    # This solves the problem of having no rivers when two long peak-saddle ridges
    # are close together and nearly parallel. Withour virtual points, Voronoi cells of the 
    # ridge1 peak/saddle might intersect ridge2 and no rivers are created inbetween
    virtualCentroids = []
    if virtualRidgePointsDist:
        for s,[p1,p2] in enumerate(saddlePeaks):
            vpts = getVirtualRidgePoints(peakCoords[p1], saddleCoords[s], virtualRidgePointsDist, 0.25)
            for p in vpts:
                virtualCentroids.append(p)
            vpts = getVirtualRidgePoints(peakCoords[p2], saddleCoords[s], virtualRidgePointsDist, 0.25)
            for p in vpts:
                virtualCentroids.append(p)
    
    if len(virtualCentroids) > 0:
        vorCentroids = np.vstack([peakCoords, saddleCoords, np.array(virtualCentroids)])
    else:
        vorCentroids = np.vstack([peakCoords, saddleCoords])
    
    # voronoi diagram, clip cells which extend to infinity
    t0 = time.perf_counter()
    vor = Voronoi(vorCentroids)
    clippedRegions, clippedVertices, vorEdgesRegions = voronoi_finite_polygons_2d(vor)
    debugInfo['timings'].append(('Voronoi', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])

    
    # 2. Build the river network
    
    # coarse river net 
    t0 = time.perf_counter()
    riverMaxElev, riverFlowTo, riverSources, riverEnds, riverDrainArea, RiverVertAdj = \
        buildVoronoiRivers(clippedVertices, clippedRegions, vorEdgesRegions,
                           peakCoords, peakElevs, saddleCoords, saddleElevs, saddlePeaks, bbox, terrainSize)
    debugInfo['timings'].append(('Coarse rivers', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])
    
    # coarse river elevs from downstream flow
    t0 = time.perf_counter()
    
    riverMaxElev[riverEnds] = minTerrainElev
    RiverVertDist = cdist(clippedVertices, clippedVertices, 'euclidean')
    
    riverElevs = getRiverHeights(riverMaxElev, riverSources, riverFlowTo, riverDrainArea, clippedVertices, 
                                 bbox, slopeCoeff=maxSlopeCoeff, minRiverElev=minTerrainElev, 
                                 srcOffsetMean=srcElevRndMean, srcOffsetStd=srcElevRndStd)
                                 
    riverElevs = smoothRiverElevs(riverElevs, riverMaxElev, RiverVertAdj,  
                                  smoothIters=coarseRiverSmoothIters, 
                                  carveOnly=False, sourcesMomentum=srcElevMomentCoarse)
                                  
    riverElevs = propagateRiverFlowElev(riverSources, riverFlowTo, riverElevs)
    
    debugInfo['timings'].append(('River elevs', time.perf_counter() - t0))
    coarseRiverElevs   = riverElevs.copy()
    coarseRiverSources = riverSources.copy()
    coarseRiverFlowTo  = riverFlowTo.copy()
    coarseRiverDrainArea = riverDrainArea.copy()
    print(debugInfo['timings'][-1])
    
    
    # 3. Refine points in both networks (ridges/rivers) by splitting large segments
    
    # refine river network
    t0 = time.perf_counter()
    riverCoords, riverElevs, riverSegs, riverFlowTo, riverDrainArea, riverSources = \
        refineRiverNetwork(riverFlowTo, clippedVertices, riverElevs, riverDrainArea, riverSources, 
                           refineDistance, riversPerturbation, bbox)
    debugInfo['timings'].append(('Refine rivers', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])

    # adjacency matrix of refined river
    t0 = time.perf_counter()
    FineRiverAdj = np.zeros((riverElevs.size, riverElevs.size))
    for s in riverSegs:
        FineRiverAdj[s[0], s[1]] = FineRiverAdj[s[1], s[0]] = 1

    # smooth elevations and (optionally) smooth position
    riverElevsSmooth  = smoothRiverElevs(riverElevs, riverElevs, FineRiverAdj, 
                                         smoothIters=refinedRiverSmoothIters, carveOnly=True, sourcesMomentum=srcElevMoment)
    riverElevsSmooth  = propagateRiverFlowElev(riverSources, riverFlowTo, riverElevsSmooth)
    riverCoordsSmooth = smoothRiverPositions(riverCoords, FineRiverAdj, 
                                             smoothIters=refinedRiverSmoothPosIters, sourcesMomentum=srcCoordsMoment)
    debugInfo['timings'].append(('Smooth rivers', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])
    
    # refine the ridge network, needed for a good Delaunay triangulation
    t0 = time.perf_counter()
    psCoords = np.vstack([peakCoords, saddleCoords])
    psElevs  = np.hstack([peakElevs, saddleElevs])
    ridges = []
    for si,s in enumerate(saddlePeaks):
        ridges.append([s[0], si + numPeaks])
        ridges.append([s[1], si + numPeaks])
        
    ridgeCoords, ridgeElevs, ridgeSegs = refineRidgeNetwork(ridges, psCoords, psElevs, 
                                                            refineDistance, ridgesPerturbation, bbox)
    debugInfo['timings'].append(('Refine ridges', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])
    
    
    # 4. Fill empty space using a blue noise distribution
    
    t0 = time.perf_counter()
    
    # kd-trees to ridgelines and riverlines
    kdRivers = cKDTree(riverCoordsSmooth)
    kdRidges = cKDTree(ridgeCoords)
    
    # keep only samples in empty spaces, by removing those too close to rivers and ridges
    closestRiverDist, closestRiverIdx = kdRivers.query(poissonSamples, k=1)
    closestRidgeDist, closestRidgeIdx = kdRidges.query(poissonSamples, k=1)
    validSamples = np.logical_and(closestRiverDist > refineDistance, closestRidgeDist > refineDistance)
    poissonCoords    = poissonSamples[validSamples,:]
    closestRiverDist = closestRiverDist[validSamples]
    closestRidgeDist = closestRidgeDist[validSamples]
    closestRiverIdx  = closestRiverIdx[validSamples]
    closestRidgeIdx  = closestRidgeIdx[validSamples]
    closestRidgeElev = ridgeElevs[closestRidgeIdx]
    closestRiverElev = riverElevs[closestRiverIdx]    
    debugInfo['timings'].append(('Poisson samples', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])
    
    
    # 5. Constrained Delaunay triangulation to obtain mesh
    
    t0 = time.perf_counter()
    
    # fix ridges and rivers in the triangulation
    fixedSegments = []
    for s in ridgeSegs:
        fixedSegments.append(s)
    for s in riverSegs:
        fixedSegments.append([s[0] + ridgeCoords.shape[0], s[1] + ridgeCoords.shape[0]])

    # constrained Delaunay
    A = dict(vertices=np.vstack([ridgeCoords, riverCoordsSmooth, poissonCoords]), segments=fixedSegments)
    B = triangle.triangulate(A, 'q10aD') # q10 for minimum angle 10deg, 
                                         # D for Delaunay (might add Steiner points)
    tvers = B['vertices']
    ttris = B['triangles']

    debugInfo['timings'].append(('Delaunay mesh', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])
    
    # neighbor representation using linked lists
    t0 = time.perf_counter()
    numMeshVerts = tvers.shape[0]
    numRidges    = ridgeElevs.size
    numRivers    = riverElevsSmooth.size
    numNetVerts  = numRidges + numRivers
    numPoisson   = numMeshVerts - numNetVerts
    # note that we only add the neighbors of valley points (poisson or steiner)
    N = [[] for _ in range(numMeshVerts)]
    for tri in ttris:
        if tri[0] >= numNetVerts:
            N[tri[0]].append(tri[1])
        if tri[1] >= numNetVerts:
            N[tri[1]].append(tri[2])
        if tri[2] >= numNetVerts:
            N[tri[2]].append(tri[0])
    debugInfo['timings'].append(('Neighbors', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])
    
    
    # 6. Elevation of the valley points
    t0 = time.perf_counter()
    closestRidgeDist = closestRidgeDist
    closestRiverDist = closestRiverDist

    # Poisson samples elevation 
    if useDrainageForValleys:
        # valley width dependant on drainage area
        closestRidgeElev = ridgeElevs[closestRidgeIdx]
        closestRiverElev = riverElevs[closestRiverIdx]
        valleyWidth = closestRidgeDist + closestRiverDist
        
        ridgeSteep  = (closestRidgeElev - peakElevs.min())/(peakElevs.max() - peakElevs.min())
        riverFactor = riverDrainArea[closestRiverIdx]**0.4 
                   
        riverWidth  = maxRiverWidth * riverFactor/riverFactor.max() * valleyWidth  
        slopeWidth  = valleyWidth - riverWidth
            
        interpCoeff  = np.minimum(1, closestRidgeDist/slopeWidth)
        poissonElevs = interpCoeff*closestRiverElev + (1 - interpCoeff)*closestRidgeElev
        
    else:
        # set sample elevation by interpolating between closest ridge and river elevations
        interpCoeff  = closestRidgeDist/(closestRidgeDist + closestRiverDist)
        poissonElevs = interpCoeff*closestRiverElev + (1 - interpCoeff)*closestRidgeElev
        
    telev = np.concatenate([ridgeElevs, riverElevsSmooth, poissonElevs])
    
    # compute elevation of the added Steiner points (if any)
    if telev.size != tvers.shape[0]:
        kdall = cKDTree(tvers[:telev.size])
        for i in range(telev.size, tvers.shape[0]):
            neighDist, neighIdx = kdall.query(tvers[i], k=5)
            ne = telev[neighIdx]
            nw = 1/neighDist
            e = np.sum(nw*ne)/np.sum(nw)
            telev = np.concatenate([telev, [e]])
            
    # finally, a bit of smoothing to avoid sharp crests (due to closest river change between neighbors)
    smoothPoissonElevIters = 2
    momentum = 0.5
    for _ in range(smoothPoissonElevIters):
        for i in range(numNetVerts, numMeshVerts):
            if len(N[i]) > 0:
                telev[i] = momentum*telev[i] + (1 - momentum)*np.mean(telev[N[i]])
                
    debugInfo['timings'].append(('Valley elevs', time.perf_counter() - t0))
    print(debugInfo['timings'][-1])
    
    
    ii,jj = RiverVertAdj.nonzero()
    riverLines = [LineString([clippedVertices[i], clippedVertices[j]]) for i,j in zip(ii,jj) if i < j]
        
    debugInfo['voronoiRegions'] = clippedRegions
    debugInfo['voronoiVerts'] = clippedVertices
    debugInfo['coarseRiverLines'] = riverLines
    debugInfo['coarseRiverElevs'] = coarseRiverElevs
    debugInfo['coarseRiverFlowTo'] = coarseRiverFlowTo
    debugInfo['coarseRiverSources'] = coarseRiverSources
    debugInfo['coarseRiverDrainArea'] = coarseRiverDrainArea
    
    #debugInfo['coarseMeshVerts'] = np.array(coarseMeshVerts)
    #debugInfo['coarseMeshElevs'] = np.array(coarseMeshElevs)
    #debugInfo['coarseMeshTris'] = np.array(coarseMeshTris)
    
    debugInfo['riverSources'] = riverSources
    debugInfo['riverFlowTo'] = riverFlowTo
    debugInfo['riverDrainArea'] = riverDrainArea
    
    debugInfo['ridgeSegments'] = ridgeSegs
    
    debugInfo['numRidgeVerts'] = ridgeElevs.size
    debugInfo['numRiverVerts'] = riverElevsSmooth.size
    debugInfo['numPoissonVerts'] = poissonElevs.size
    
    debugInfo['closestRidgeIdx'] = closestRidgeIdx
    debugInfo['closestRiverIdx'] = closestRiverIdx
    
    
    return tvers, telev, ttris, debugInfo