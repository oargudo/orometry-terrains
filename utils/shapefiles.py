import shapefile  # http://github.com/GeospatialPython/pyshp
from shapely.geometry import Point, Polygon, MultiPolygon
from .coords import *


def polygonFromShapefile(shapeFilePath):
    # process shapefile, read the polygon shape of the region
    polys = []
    sf = shapefile.Reader(shapeFilePath)
    for shapeRec in sf.shapeRecords():
        # a shape can be divided in several polygonal parts
        numParts = len(shapeRec.shape.parts)
        for i in range(numParts):
            ini = shapeRec.shape.parts[i]
            if i+1 < numParts:
                end = shapeRec.shape.parts[i+1]
                polys.append(Polygon(shapeRec.shape.points[ini:end]))
            else:
                polys.append(Polygon(shapeRec.shape.points[ini:]))
    regionPoly = MultiPolygon(polys)
    
    return regionPoly
    
    
def sampleShapefileLocations(shapeFilePath, diskRadius):
    
    # read shapefile polygon
    regionPoly = polygonFromShapefile(shapeFilePath)
    
    # erode the radius
    regionPoly = regionPoly.buffer(-km2deg(diskRadius))
    
    # sample stats locations inside polygon
    minLon, minLat, maxLon, maxLat = regionPoly.bounds
    sampleLocations = []
    
    # latitude step is constant
    stepSizeLat = km2deg(diskRadius)
    glat = minLat + stepSizeLat*0.25
    while glat <= maxLat:
        # longitude step depends on latitude
        stepSizeLon = km2deg(diskRadius, glat)
        glon = minLon + stepSizeLon*0.25
        while glon <= maxLon:
            # perturb grid position by 1/4 of grid size (minimum possible distance will be 1/2 grid size)
            slat = glat + np.random.uniform(-1, 1)*stepSizeLat*0.25
            slon = glon + np.random.uniform(-1, 1)*stepSizeLon*0.25
            
            # check if location is inside polygon
            if regionPoly.contains(Point(slon, slat)):
                sampleLocations.append((slat, slon))
                
            glon += stepSizeLon
        glat += stepSizeLat
    
    return sampleLocations