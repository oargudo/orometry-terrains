import argparse
import os
from math import sqrt
import random
import numpy as np
from scipy.spatial import cKDTree


parser = argparse.ArgumentParser()
parser.add_argument("isolationList", help="isolation file")
parser.add_argument("prominenceList", help="prominence file")
parser.add_argument("outFile", help="output file")
parser.add_argument("--fileHeaders", help="files contain headers", action='store_true')
parser.add_argument("--deleteOriginals", help="delete isolation and prominence list files", action='store_true')
args = parser.parse_args()


fiso = open(args.isolationList)
if args.fileHeaders:
    fiso.readline()
fpro = open(args.prominenceList)
if args.fileHeaders:
    fpro.readline()

fout = open(args.outFile, 'w')
fout.write('latitude,longitude,elevation in feet,key saddle latitude,key saddle longitude,prominence in feet,isolation latitude,isolation longitude,isolation in km\n')

print('Reading isolations')
isos = []
ni = 0
for line in fiso:
    isos.append([float(x) for x in line.split(',')])
    ni += 1
    
isoPoints = np.zeros((ni, 2))
for i in range(ni):
    isoPoints[i,0] = isos[i][0]
    isoPoints[i,1] = isos[i][1]
kdIsos = cKDTree(isoPoints)
    
print('Merging isolation and prominence lists')
numPeaks = 0
numIsolated = 0
for line in fpro:
    vals = [float(x) for x in line.split(',')]
    minDist = (3/3600)*3.3 # about 300m
    match = None
    
    nndist, nnid = kdIsos.query(np.array([vals[0], vals[1]]), k=1)
    if nndist < 0.25*minDist or nndist < minDist and abs(vals[2] - isos[nnid][2]) < 200:
        match = isos[nnid]
        
    if match:
        fout.write('%s,%.4f,%.4f,%.4f\n' % (line.strip(), match[3], match[4], match[5]))
    else:
        fout.write('%s,%.4f,%.4f,%.4f\n' % (line.strip(), vals[0] + 3/3600*(random.random()*2-1), vals[1] + 3/3600*(random.random()*2-1), 0.1))
        numIsolated += 1
    
    numPeaks += 1
    
    
fiso.close()
fpro.close()
fout.close()

print('Processed %d peaks, low isolation on %d' %( numPeaks, numIsolated))

if args.deleteOriginals:
    os.remove(args.isolationList)
    os.remove(args.prominenceList)
