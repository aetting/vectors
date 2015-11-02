#!/usr/bin/python
# -*- coding: utf8 -*-


##python evaluateSim.py /Users/allysonettinger/Desktop/vectors/polySyn/enModelg.sense /Users/allysonettinger/Desktop/similarity-datasets/MEN/MEN_dataset_natural_form_full

import numpy, scipy, gzip, sys, gensim, re
from scipy import stats, spatial

oov = {}

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
    
def avgSimPair(w1,w2,vecDict,senSum,wordTot):
    w1list = [k for k in vecDict if k.split('%')[0] == w1] 
    w2list = [k for k in vecDict if k.split('%')[0] == w2]
    
#     print w1list
#     print w2list
    if len(w1list) == 0: oov[w1] = 1
    if len(w2list) == 0: oov[w2] = 1
    simSum = 0
    maxSim = 0
    normalizer = float(len(w1list)*len(w2list))
    print 'lengths:' + str(len(w1list)) + ' ' + str(len(w2list))
    
    if normalizer == 0: return None 
    senSum += (len(w1list) + len(w2list))
    wordTot += 2
    
    '''iterate over senses'''
    for w1 in w1list:
        for w2 in w2list:
            sim = cosSim(vecDict[w1],vecDict[w2])
            simSum += sim
            if sim > maxSim: 
                maxSim = sim
                maxpair = (w1,w2)
            
    avgSim = simSum/normalizer
    
    return (avgSim, maxSim, maxpair,senSum, wordTot)
    
def getSpearman(vectorDict,simSet):
    vecSimsAvg = []
    vecSimsMax = []
    simSetSims = []
    senSum = 0
    wordTot = 0
    for w1,w2,s in simSet:
        w1 = w1.lower()
        w2 = w2.lower()
        m1 = re.match('([a-z]+)\-[a-z]',w1)
        m2 = re.match('([a-z]+)\-[a-z]',w2)
        if m1: w1 = m1.group(1)
        if m2: w2 = m2.group(1)
        vecSim = avgSimPair(w1,w2,vectorDict,senSum,wordTot)
        if vecSim is None: 
            print '\n' + w1 + ' or ' + w2 + ' missing!\n'
            continue
        vecSimAvg = vecSim[0]
        vecSimMax = vecSim[1]
        
        senSum = vecSim[3]
        wordTot = vecSim[4]
        
        simSetSims.append(float(s))
        vecSimsAvg.append(vecSimAvg)
        vecSimsMax.append(vecSimMax)
        
        print s
        print vecSim[:2]
        print ' '.join([str(e) for e in vecSim[2]]) 
        
        
    rho,p = scipy.stats.spearmanr(vecSimsAvg,simSetSims)
    rho2,p2 = scipy.stats.spearmanr(vecSimsMax,simSetSims)
    print 'RHO (avg): ' + str(rho)
    print 'RHO (max): ' + str(rho2)
    print 'Avg # senses: ' + str(senSum/float(wordTot))
    print oov
    return rho, rho2,p

def iterSimSets(vectorFile, genFormat, simSetFiles):

    if genFormat == 'gen': vectorDict = loadVectors(vectorFile)
    elif genFormat == 'text': vectorDict = readVectors(vectorFile)
    else:
        sys.stderr.write('Specify format: \'text\' or \'gen\'\n') 
        return
    rhoList = []
    for set in simSetFiles:
        simSet = numpy.loadtxt(set,dtype='str')
        rho,rho2,p = getSpearman(vectorDict,simSet)
        rhoList.append((set,rho,rho2))
        
    for item in rhoList: print item
    print vectorFile
    
        
if __name__ == "__main__":
    iterSimSets(sys.argv[1],sys.argv[2],sys.argv[3:])
#     getVectors(sys.argv[1])