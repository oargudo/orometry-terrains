from noise import snoise2
import numpy as np


def getNoise(x, y, seed=0, scale=1, octaves=8):
    return snoise2(scale*x + seed, scale*y, octaves=octaves)

def getNoiseTexture(size, seed=0, scale=1):    
    tex = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            y = scale*i/size[0] + seed
            x = scale*j/size[1]
            tex[i,j] = snoise2(x, y, octaves=8)
    return tex   