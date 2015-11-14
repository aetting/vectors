#!/usr/bin/python
# -*- coding: utf8 -*-


##python evaluateSim.py /Users/allysonettinger/Desktop/vectors/polySyn/enModelg.sense /Users/allysonettinger/Desktop/similarity-datasets/MEN/MEN_dataset_natural_form_full

import numpy, scipy, gzip, sys, gensim, re, os, getopt
from scipy import stats, spatial

oov = {}

def cosSim(u,v):
    return (1 - scipy.spatial.distance.cosine(u,v))

def readCommandLineInput(argv):
    try:
        #specify the possible option switches
        opts, _ = getopt.getopt(sys.argv[1:], "v:s:g:y:", ["vectors=","set=","genformat=,sync="])
    except: print 'INPUT INCORRECT'
    vecList = []
    setList = []
    genFormatList = []
    sync = None
    # option processing
    for option, value in opts:
        if option in ("-v", "--vectors"):
            vecList.append(value)
        elif option in ("-s", "--set"):
            setList.append(value)
        elif option in ("-g", "--genformat"):
            genFormatList.append(value)
        elif option in ("-y", "--sync"):
            sync = bool(int(value))
        else:
            print "Doesn't match any option"
    if len(vecList) != len(genFormatList):
        print "Need equal number of vector files and format specifications!"
        sys.exit()
    print (vecList,setList,genFormatList,sync)
    return (vecList,setList,genFormatList,sync)

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
    
def getPhraseMax(p1,p2,vecDict,out):
    out.write('PROCESSING PHRASE'+'\n')
    maxSim = 0
    maxPair = (None,None)

    for w1 in p1.split():
        for w2 in p2.split():
            simSum = 0
            w1list = [k for k in vecDict if k.split('%')[0] == w1] 
            w2list = [k for k in vecDict if k.split('%')[0] == w2]
#             out.write(' '.join(w1list) + '\n')
#             out.write(' '.join(w2list) + '\n')
            normalizer = float(len(w1list)*len(w2list))
            if normalizer == 0: 
                out.write(w1 + ' or ' + w2 + ' missing'+'\n')
                continue
            for w1l in w1list:
                for w2l in w2list:
                    sim = cosSim(vecDict[w1l],vecDict[w2l])
                    simSum += sim
            avgSim = simSum/normalizer
            out.write('PAIR: ' + w1 + ' + ' + w2 +'\n')
            if avgSim > maxSim:
                maxSim = avgSim
                maxPair = (w1,w2)
    out.write('TOP PAIR = ' + str(maxPair)+'\n')
    return maxPair
    
def iterPairSenses(w1,w2,vecDict,out):

    if len(w1.split()) > 1 or len(w2.split()) > 1:
        w1,w2 = getPhraseMax(w1,w2,vecDict,out)
    if not w1 or not w2: return (0,0,0,0,0)
    
    w1list = [k for k in vecDict if k.split('%')[0] == w1] 
    w2list = [k for k in vecDict if k.split('%')[0] == w2]
    
#     out.write(' '.join(w1list) + '\n')
#     out.write(' '.join(w2list) + '\n')
    if len(w1list) == 0: 
        oov[w1] = 1
        out.write(w1 + ' missing'+'\n')
    if len(w2list) == 0: 
        oov[w2] = 1
        out.write(w2 + ' missing'+'\n')
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
            out.write('PAIR: ' + w1 + ' + ' + w2 +'\n')
            simSum += sim
            if sim > maxSim: 
                maxSim = sim
                maxPair = (w1,w2)
            
    avgSim = simSum/normalizer
    
    return (maxSim,maxPair,avgSim,len(w1list),len(w2list))
    
def getSynAccuracy(vectorDict,synSetFile,vecName,setName,vectorFile,sync):
    numCorr = 0
    numCounted = 0
    numLines = 0
    skiplines = 0
    synSet = open(synSetFile)
    m = re.match('.+/([^/]+)$',vecName)
    vecName_trunc = m.group(1)
    if sync: ss = '-sync'
    else: ss = '-raw'
    outFile = open(vectorFile+'-'+setName+ss+'-breakdown','w')
    outFile3 = open(vectorFile+'-'+setName+ss+'-fulloutput','w')
    lineInd = -1
    summaryList = []
    probeList = []
    for l in synSet:
        numLines += 1
        lineInd += 1
        line = l.strip().split(' | ')
        maxSimCounter = 0
        probe = line[0].lower().strip()
        probeList.append(probe)
        corr = line[1].lower().strip()
        options = line[1:]
        outFile3.write('PROBE: ' + probe+ '\n')
        outFile3.write('CORRECT: ' + corr+ '\n')
        for op in options:
            op = op.lower().strip()
            outFile3.write(op+ '\n')
            maxSim,maxPair,avgSim,prlen,oplen = iterPairSenses(probe,op,vectorDict,outFile3)
            if prlen == 0 or oplen == 0: 
                winner = None
                break
            outFile3.write(str(maxSim) + ': ' + ' '.join(maxPair)+ '\n')
            if maxSim > maxSimCounter:
                maxSimCounter = maxSim
                winner = op
        if not winner: 
            outFile3.write(probe + ' line skipped\n')
            skiplines += 1
            outFile.write('	'.join([probe,corr]))
            outFile.write('	SKIPPED\n')
            summaryList.append(None)
            continue
        outFile3.write('WINNER: ' + winner + '\n\n')
        outFile.write('	'.join([probe,corr,winner]))
        numCounted += 1
        if winner == corr: 
            numCorr += 1
            outFile.write('	1\n')
            summaryList.append(1)
        else: 
            outFile.write('	0\n')
            summaryList.append(0)
    
    outFile.write(str(skiplines) + ' lines skipped\n')
    outFile.write(vecName)
    outFile.close()
    synSet.close()    
    acc = (100*numCorr/float(numLines))
    outFile3.write('ACC: ' + str(acc)+ '\n')
    outFile3.write('Num Corr: ' + str(numCorr)+ '\n')
    outFile3.write('Num Counted: ' + str(numCounted)+ '\n')
    outFile3.write(str(skiplines) + ' LINES SKIPPED FROM ' + setName+ '\n\n')
    outFile3.close()
    return acc,skiplines,summaryList,probeList
    
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
        acc,skipped,summaryList = getSynAccuracy(vectorDict,set,vecName,setName,vectorFile)
        accList.append((setName,acc,str(skipped)+' skipped'))
        
    print 'SYN RESULTS'
    for item in accList: print item
    print vecName
    print oov
    
    
def syncSynSets(vectorList, genFormatList, synSetFiles, sync):

    vecDictList = []
    for i in range(len(vectorList)):  
        if genFormatList[i] == 'gen': 
            vecDictList.append(loadVectors(vectorList[i]))
        elif genFormatList[i] == 'text': 
            vecDictList.append(readVectors(vectorList[i]))
        else:
            sys.stderr.write('Specify format: \'text\' or \'gen\'\n') 
            sys.exit()
    accListsAll = []
    for set in synSetFiles:
        ms = re.match('.+/([^/]+)$',set)
        setName = ms.group(1)
        if sync:
            out = open(vectorList[0]+'-'+setName+'-sync-full-summary.csv','w')
            out2 = open(vectorList[0]+'-'+setName+'-sync-overlap-summary.csv','w')
        allLists = []
        accList = []
        for i in range(len(vectorList)):
            vectorFile = vectorList[i]
            vectorDict = vecDictList[i]
            m = re.match('.+/([^/]+/[^/]+)$',vectorFile)
            vecName = m.group(1)
            if sync:
                out.write(vecName + ',')
                out2.write(vecName + ',')
            acc,skipped,summaryList,probeList = getSynAccuracy(vectorDict,set,vecName,setName,vectorFile,sync)
            if not sync:
                accList.append((setName,vecName,acc,str(skipped)+' skipped'))
            else:
                allLists.append(summaryList)
        if sync:
            out.write('\n')
            toSkip = []
            for j in range(len(allLists[0])):
                skip = 0
                for list in allLists:
                    if list[j] is None: skip = 1
                if skip: toSkip.append(j)
                out.write(probeList[j] + ',')
                out.write(','.join([str(list[j]) for list in allLists])+'\n')
                if not skip: 
                    out2.write(probeList[j] + ',')
                    out2.write(','.join([str(list[j]) for list in allLists])+'\n')
            print 'TOSKIP'
            print setName
            print toSkip
            syncSkipped = len(toSkip)
            for i in range(len(vectorList)):
                syncedCorrList = [allLists[i][n] for n in range(len(allLists[0])) if n not in toSkip]
                norm = float(len(syncedCorrList))
                print norm
                accSync = sum(syncedCorrList)/norm
                m = re.match('.+/([^/]+/[^/]+)$',vectorList[i])
                vecName = m.group(1)
                accList.append((setName,vecName,accSync,str(syncSkipped)+' skipped'))      
                
        accListsAll.append(accList)
        
        if sync:
            out.close()
            out2.close()
        
    print 'SYN RESULTS'
    for list in accListsAll: 
        for item in list: print item
    print oov
    
        
if __name__ == "__main__":
    vecList,setList,genFormat,sync = readCommandLineInput(sys.argv)
    syncSynSets(vecList,genFormat,setList,sync)
