import numpy as np
import math
from math import sqrt, radians, cos, sin, asin, atan2, fmod


# 1 arcsec ~= 30.87 m (at equator)
def km2deg(x, latitude=0):
    return x/(3.6*30.87*np.cos(np.radians(latitude))) 

def deg2km(x, latitude=0):
    return x*3.6*30.87*np.cos(np.radians(latitude))

def feet2m(x):
    return x*0.3048

def m2feet(x):
    return x*3.28084

def distance_haversine(p1, p2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)

    Haversine formula: 
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 · atan2( √a, √(1−a) )
        d = R · c

    where φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);
            note that angles need to be in radians to pass to trig functions!

    :p1:     (tup) lat,lon
    :p2:     (tup) lat,lon
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    R = 6371.0 # km - earths's radius
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) # 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c
    return d

def angle_direction(p1, p2):
    lat1, lon1 = p1
    lat2, lon2 = p2
    dlat = deg2km(lat2 - lat1)
    dlon = deg2km(lon2 - lon1, 0.5*(lat1+lat2))
    dir = 180.0*atan2(dlat, dlon)/math.pi
    if dir < 0:
        dir += 360.0
    return dir