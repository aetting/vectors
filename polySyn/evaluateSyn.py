#!/usr/bin/python
# -*- coding: utf8 -*-


##python evaluateSim.py /Users/allysonettinger/Desktop/vectors/polySyn/enModelg.sense /Users/allysonettinger/Desktop/similarity-datasets/MEN/MEN_dataset_natural_form_full

import numpy, scipy, gzip, sys, gensim, re, os
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
    
def getPhraseMax(p1,p2,vecDict):
    print 'PROCESSING PHRASE'
    maxSim = 0
    maxPair = (None,None)

    for w1 in p1.split():
        for w2 in p2.split():
            simSum = 0
            w1list = [k for k in vecDict if k.split('%')[0] == w1] 
            w2list = [k for k in vecDict if k.split('%')[0] == w2]
            normalizer = float(len(w1list)*len(w2list))
            if normalizer == 0: 
                print w1 + ' or ' + w2 + ' missing'
                continue
            for w1l in w1list:
                for w2l in w2list:
                    sim = cosSim(vecDict[w1l],vecDict[w2l])
                    simSum += sim
            avgSim = simSum/normalizer
            print 'PAIR: ' + w1 + ' + ' + w2 
            if avgSim > maxSim:
                maxSim = avgSim
                maxPair = (w1,w2)
    print 'TOP PAIR = ' + str(maxPair)
    return maxPair
    
def iterPairSenses(w1,w2,vecDict):

    if len(w1.split()) > 1 or len(w2.split()) > 1:
        w1,w2 = getPhraseMax(w1,w2,vecDict)
    if not w1 or not w2: return (0,0,0,0,0)
    
    w1list = [k for k in vecDict if k.split('%')[0] == w1] 
    w2list = [k for k in vecDict if k.split('%')[0] == w2]
    
#     print w1list
#     print w2list
    if len(w1list) == 0: 
        oov[w1] = 1
        print w1 + ' missing'
    if len(w2list) == 0: 
        oov[w2] = 1
        print w2 + ' missing'
    simSum = 0
    maxSim = 0
    maxPair = ()
    normalizer = float(len(w1list)*len(w2list))
    if normalizer == 0: return (0,0,0,0,0)
#     print 'lengths:' + str(len(w1list)) + ' ' + str(len(w2list)) 
    
    '''iterate over senses'''
    for w1 in w1list:
        for w2 in w2list:
            sim = cosSim(vecDict[w1],vecDict[w2])
            print 'PAIR: ' + w1 + ' + ' + w2 
            simSum += sim
            if sim > maxSim: 
                maxSim = sim
                maxPair = (w1,w2)
            
    avgSim = simSum/normalizer
    
    return (maxSim,maxPair,avgSim,len(w1list),len(w2list))
    
def getSynAccuracy(vectorDict,synSetFile,vecName,setName,vectorFile):
    numCorr = 0
    numCounted = 0
    numLines = 0
    skiplines = 0
    synSet = open(synSetFile)
    m = re.match('.+/([^/]+)$',vecName)
    vecName_trunc = m.group(1)
    outFile = open(vectorFile+'-'+setName+'-breakdown','w')
    for l in synSet:
        numLines += 1
        line = l.strip().split(' | ')
        maxSimCounter = 0
        probe = line[0].lower().strip()
        corr = line[1].lower().strip()
        options = line[1:]
        print 'PROBE: ' + probe
        print 'CORRECT: ' + corr
        for op in options:
            op = op.lower().strip()
            print op
            maxSim,maxPair,avgSim,prlen,oplen = iterPairSenses(probe,op,vectorDict)
            if prlen == 0 or oplen == 0: 
                winner = None
                break
            print str(maxSim) + ': ' + str(maxPair)
            if maxSim > maxSimCounter:
                maxSimCounter = maxSim
                winner = op
        if not winner: 
            print probe + ' line skipped\n'
            skiplines += 1
            outFile.write('	'.join([probe,corr]))
            outFile.write('	SKIPPED\n')
            continue
        print 'WINNER: ' + winner + '\n'
        outFile.write('	'.join([probe,corr,winner]))
        numCounted += 1
        if winner == corr: 
            numCorr += 1
            outFile.write('	1\n')
        else: outFile.write('	0\n')
    
    outFile.write(str(skiplines) + ' lines skipped\n')
    outFile.write(vecName)
    outFile.close()
    synSet.close()    
    acc = (100*numCorr/float(numLines))
    print 'ACC: ' + str(acc)
    print 'Num Corr: ' + str(numCorr)
    print 'Num Counted: ' + str(numCounted)
    print str(skiplines) + ' LINES SKIPPED FROM ' + setName
    print '\n\n'
    return acc,skiplines
    
def iterSynSets(vectorFile, genFormat, synSetFiles):

    if genFormat == 'gen': vectorDict = loadVectors(vectorFile)
    elif genFormat == 'text': vectorDict = readVectors(vectorFile)
    else:
        sys.stderr.write('Specify format: \'text\' or \'gen\'\n') 
        return
    accList = []
    m = re.match('.+/([^/]+/[^/]+)$',vectorFile)
    vecName = m.group(1)
    for set in synSetFiles:
        m = re.match('.+/([^/]+)$',set)
        setName = m.group(1)
        acc,skipped = getSynAccuracy(vectorDict,set,vecName,setName,vectorFile)
        accList.append((setName,acc,str(skipped)+' skipped'))
        
    print 'SYN RESULTS'
    for item in accList: print item
    print vecName
    print oov
    
        
if __name__ == "__main__":
    iterSynSets(sys.argv[1],sys.argv[2],sys.argv[3:])
#     getVectors(sys.argv[1])