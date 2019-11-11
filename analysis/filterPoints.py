import argparse
import shapefile  # http://github.com/GeospatialPython/pyshp
import shapely    # http://toblerity.org/shapely/manual.html
from shapely.geometry import Point, Polygon, MultiPolygon


parser = argparse.ArgumentParser()
parser.add_argument("shapesFile", help=".shp with the polygons of the regions")
parser.add_argument("pointsFile", help=".txt file with the points to check")
parser.add_argument("outFile", help="output file")
parser.add_argument("--tolerance", help="distance (m) around polygon test", default=0, type=float)
args = parser.parse_args()


sf = shapefile.Reader(args.shapesFile)
with open(args.shapesFile) as f:
    sfencoding = f.encoding

pointsfile = open(args.pointsFile, encoding='utf-8')
xcol = 1
ycol = 0

outfile = open(args.outFile, 'w')

numline = 1
numPeaks = 0

# print(sf.fields)

# iterate over shape-records
for shapeRec in sf.shapeRecords():

    # a shape can be divided in several polygonal parts
    numParts = len(shapeRec.shape.parts)
    polys = []
    for i in range(numParts):
        ini = shapeRec.shape.parts[i]
        if i+1 < numParts:
            end = shapeRec.shape.parts[i+1]
            polys.append(Polygon(shapeRec.shape.points[ini:end]))
        else:
            polys.append(Polygon(shapeRec.shape.points[ini:]))
            
    # create a multipolygon with all the individual parts
    p = MultiPolygon(polys)
    if args.tolerance != 0:
        p = p.buffer(args.tolerance)
        
    # find which points fall inside the polygon
    try:
        for pline in pointsfile:
            vals = pline.split(',')
            if p.contains(Point([float(vals[xcol]), float(vals[ycol])])):
                outfile.write(pline.encode('utf-8', errors='replace').decode('utf-8'))
                numPeaks += 1
            numline += 1
    except Exception as e:        
        print('Aborting execution at line %d'%numline)
        print('Error:', e)
        
print('Found %d peaks'%numPeaks)
