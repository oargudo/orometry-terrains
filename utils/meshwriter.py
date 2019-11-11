import numpy as np
from shapely.geometry import Polygon, Point


def writeTerrainMesh(objName, meshVerts, meshElevs, meshTris, bbox):

    # crop to bbox
    numTris = meshTris.shape[0]
    trisInside = [False for _ in range(numTris)]
    for ti in range(numTris):

        tri = meshTris[ti]
        triVerts = meshVerts[tri,:]
        poly = Polygon(triVerts)

        if poly.overlaps(bbox):

            xpoly  = poly.intersection(bbox).exterior
            vertId = [-1 for _ in range(len(xpoly.coords)-1)]

            for pi,p in enumerate(triVerts):
                v = Point(p)
                if bbox.contains(v):
                    vertId[np.argmin([v.distance(Point(x)) for x in xpoly.coords])] = tri[pi]

            for pi in range(len(vertId)):
                if vertId[pi] < 0:
                    vertId[pi] = len(meshVerts)
                    meshVerts = np.vstack([meshVerts, [xpoly.coords[pi][0], xpoly.coords[pi][1]]])
                    meshElevs = np.hstack([meshElevs, np.array([0])])

            meshTris[ti] = [vertId[0], vertId[1], vertId[2]]
            for vi in range(3,len(vertId)):
                meshTris = np.vstack([meshTris, [vertId[0], vertId[vi-1], vertId[vi]]])

        else:
            trisInside[ti] = True

    # clean unused verts
    vertInMesh = np.full((meshVerts.shape[0],), False)
    vertInMesh[meshTris.flatten()] = True
    vertNewIdx = np.cumsum(vertInMesh)    
    
    # write obj
    fout = open(objName, 'w')
    fout.write('# obj file\n')
    for i,v in enumerate(meshVerts):
        if vertInMesh[i]:
            fout.write('v %.2f %.5f %.5f\n' % (np.clip(v[0], bbox.bounds[0], bbox.bounds[2]), 
                                               np.clip(v[1], bbox.bounds[1], bbox.bounds[3]), 
                                               0.001*meshElevs[i]))
    for i,tri in enumerate(meshTris):
        p0 = np.array([meshVerts[tri[0]][0], meshVerts[tri[0]][1], 0])
        p1 = np.array([meshVerts[tri[1]][0], meshVerts[tri[1]][1], 0])
        p2 = np.array([meshVerts[tri[2]][0], meshVerts[tri[2]][1], 0])
        if np.cross(p1-p0, p2-p0)[2] > 0:
            fout.write('f %d %d %d\n' % (vertNewIdx[tri[0]], vertNewIdx[tri[1]], vertNewIdx[tri[2]]))
        else:
            fout.write('f %d %d %d\n' % (vertNewIdx[tri[2]], vertNewIdx[tri[1]], vertNewIdx[tri[0]]))

    fout.close()