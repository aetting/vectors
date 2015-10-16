#!/usr/bin/python
# -*- coding: utf8 -*-


##python evaluateSim.py /Users/allysonettinger/Desktop/vectors/polySyn/enModelg.sense /Users/allysonettinger/Desktop/similarity-datasets/MEN/MEN_dataset_natural_form_full

import numpy, scipy, gzip, sys, gensim
from scipy import stats, spatial

def cosSim(u,v):
    return (1 - scipy.spatial.distance.cosine(u,v))

def readVectors(filename):
    if filename.endswith('.gz'):
        fileObject = gzip.open(filename, 'r')
    else:
        fileObject = open(filename, 'r')
    
    vectorDim = int(fileObject.readline().strip().split()[1])
    vectors = numpy.loadtxt(filename, dtype=float, comments=None, skiprows=1, usecols=range(1,vectorDim+1))
    
    print len(vectors)
    
    wordVectors = {}
    lineNum = 0
    for line in fileObject:
        word = line.lower().strip().split()[0]
        wordVectors[word] = vectors[lineNum]
        lineNum += 1
    
    sys.stderr.write('Finished reading vectors.\n')
    
    fileObject.close()
    print len(wordVectors)
    
    return wordVectors

def loadVectors(filename):
    sys.stderr.write('Loading vectors from array...\n')
    
    genVectors = gensim.models.Word2Vec.load(filename) 
       
    wordVectors = {k:genVectors[k] for k in genVectors.vocab}
    vectorDim = len(wordVectors['the'])
    
    print len(wordVectors)
    
    sys.stderr.write('Finished reading vectors.\n')
    
    return wordVectors
    
def avgSimPair(w1,w2,vecDict):
    w1list = [k for k in vecDict if k.split('%')[0] == w1] 
    w2list = [k for k in vecDict if k.split('%')[0] == w2]
    
#     print w1list
#     print w2list
    simSum = 0
    normalizer = float(len(w1list)*len(w2list))
    print 'lengths:' + str(len(w1list)) + ' ' + str(len(w2list))
    
    if normalizer == 0: return None
    
    '''iterate over senses'''
    for w1 in w1list:
        for w2 in w2list:
            sim = cosSim(vecDict[w1],vecDict[w2])
            simSum += sim
            
    avgSim = simSum/normalizer
    
    return avgSim

def compareSimSets(vectorFile,simSetFile,genFormat):

    if genFormat == 'gen': vectorDict = loadVectors(vectorFile)
    elif genFormat == 'text': vectorDict = readVectors(vectorFile)
    else:
        sys.stderr.write('Specify format: \'text\' or \'gen\'\n') 
        return
    simSet = numpy.loadtxt(simSetFile,dtype='str')
    
    vecSims = []
    simSetSims = []
    
    for w1,w2,s in simSet:
        vecSim = avgSimPair(w1,w2,vectorDict)
        if vecSim is None: continue
        
        simSetSims.append(float(s))
        vecSims.append(vecSim)
        
        print s
        print vecSim
        
    rho,p = scipy.stats.spearmanr(vecSims,simSetSims)
    print 'RHO ' + str(rho)
    
if __name__ == "__main__":
    compareSimSets(sys.argv[1],sys.argv[2],sys.argv[3])
#     getVectors(sys.argv[1])