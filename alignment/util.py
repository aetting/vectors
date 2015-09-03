#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy, scipy
from scipy import spatial

def readStoplist(f):
    stopwords = []
    with open(f) as stopList:
        stopText = stopList.read()
        for line in stopText.split('\n'):
            line = line.strip()
            stopwords.append(line)
    return stopwords
    
def centerScale(val,vec):
    span = max(vec)-min(vec)
    newval = (val-min(vec))/float(span)
    return newval
    
class SenseObj(object):
    def __init__(self,lemma,sense):
        self.lemma = lemma
        self.sense = sense
        
def getAvgVec(counter,numwords,vecmodel):
    top = counter.most_common(numwords)
    countList = [c[1] for c in top]
    countSum = sum(countList)
    cntxtVecs = []
    cntxtWgts = []
    for cw,ct in top:
        try: vecmodel[cw]
        except: 
            continue
        else:
            cntxtVecs.append(vecmodel[cw])
            cntxtWgts.append(float(ct)/countSum)
    if len(cntxtVecs) > 0:
        cntxtAvgVec = numpy.average(cntxtVecs,axis=0,weights=cntxtWgts)
    else:
        cntxtAvgVec = 'NaN'
    return cntxtAvgVec
    
def cosSim(u,v):
    return (1 - scipy.spatial.distance.cosine(u,v))
    
def getWindow(pos,window,listlen):
    if pos - window > 0: left = pos - window
    else: left = 0
    if pos + window <= (listlen -1): right = pos + (window + 1)
    else: right = listlen
    return left,right