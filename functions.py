# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:59:13 2013

@author: tokichie
"""

import numpy as np

'''
# @param chm Canopy Height Model
# @param cmap Crown Map
'''
def processCHM(chm, cmap):
    numOfCrowns = np.max(cmap)
    res = np.zeros((len(chm), len(chm[0])), dtype=np.float64)
    for i in range(numOfCrowns):
        mask = np.nonzero(cmap == i+1)
        res[mask] = np.max(chm[mask])
        
    return res
        