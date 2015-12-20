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
        opts, _ = getopt.getopt(sys.argv[1:], "v:s:g:c:m:", ["vectors=","set=","genformat=","combo=","combomix="])
    except: print 'INPUT INCORRECT'
    vecList = []
    setList = []
    genFormatList = []
    comboList = []
    sync = None
    mix = None
    # option processing
    for option, value in opts:
        if option in ("-v", "--vectors"):
            vecList.append(value)
        elif option in ("-s", "--set"):
            setList.append(value)
        elif option in ("-g", "--genformat"):
            genFormatList.append(value)
        elif option in ("-c", "--combo"):
            comboList.append(value)
        elif option in ("-m", "--combomix"):
            mix = value
        else:
            print "Doesn't match any option"
    if len(vecList) != len(genFormatList):
        print "Need equal number of vector files and format specifications!"
        sys.exit()
    if len(comboList) > 0:
        vecList.append(comboList)
    print (vecList,setList,genFormatList)
    return (vecList,setList,genFormatList,mix)

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
    
def avgSimPair(w1,w2,vecDict,senSum,wordTot,out):
    w1list = [k for k in vecDict if k.split('%')[0] == w1] 
    w2list = [k for k in vecDict if k.split('%')[0] == w2]
    
    out.write(' '.join(w1list) + '\n')
    out.write(' '.join(w2list) + '\n')
    if len(w1list) == 0: oov[w1] = 1
    if len(w2list) == 0: oov[w2] = 1
    simSum = 0
    maxSim = 0
    maxPair = ()
    normalizer = float(len(w1list)*len(w2list))
    out.write('lengths:' + str(len(w1list)) + ' ' + str(len(w2list))+ '\n')
    
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
                maxPair = (w1,w2)
            
    avgSim = simSum/normalizer
    
    return (avgSim, maxSim, maxPair,senSum, wordTot)
    
def getSpearman(vectorDict,simSetFile,setName,vectorFile):
    vecSimsAvg = []
    vecSimsMax = []
    simSetSims = []
    out = open(vectorFile+'-'+setName+'-fulloutput','w')
    simSet = open(simSetFile)
    senSum = 0
    wordTot = 0
    pairs_skipped = 0
    for line in simSet:
        split = line.split()
        w1 = split[0].lower()
        w2 = split[1].lower()
        out.write(w1 + ' ' + w2 + '\n')
        s = split[2]
        m1 = re.match('([a-z]+)\-[a-z]',w1)
        m2 = re.match('([a-z]+)\-[a-z]',w2)
        if m1: w1 = m1.group(1)
        if m2: w2 = m2.group(1)
        vecSim = avgSimPair(w1,w2,vectorDict,senSum,wordTot,out)
        if vecSim is None: 
            out.write('\n' + w1 + ' or ' + w2 + ' missing!\n')
            pairs_skipped += 1
            continue
        vecSimAvg = vecSim[0]
        vecSimMax = vecSim[1]
        
        senSum = vecSim[3]
        wordTot = vecSim[4]
        
        simSetSims.append(float(s))
        vecSimsAvg.append(vecSimAvg)
        vecSimsMax.append(vecSimMax)
        
        out.write(s + '\n')
        out.write(' '.join(str(e) for e in vecSim[:2]) + '\n')
        out.write('MAX: ' + ' '.join([str(e) for e in vecSim[2]])+ '\n\n')
        
        
    rho,p = scipy.stats.spearmanr(vecSimsAvg,simSetSims)
    rho2,p2 = scipy.stats.spearmanr(vecSimsMax,simSetSims)
    out.write('RHO (avg): ' + str(rho)+ '\n')
    out.write('RHO (max): ' + str(rho2)+ '\n')
    out.write('Avg # senses: ' + str(senSum/float(wordTot))+ '\n')
    out.write(str(pairs_skipped) + ' PAIRS SKIPPED FROM ' + setName+ '\n')
    out.close()
    return rho, rho2,p,pairs_skipped
    
def getComboSpearman(vectorDicts,simSetFile,setName,vectorList,mix):
    vecSimsAvg = []
    vecSimsMax = []
    simSetSims = []
    out = open(vectorList[0]+'-COMBO-'+setName+'-fulloutput','w')
    simSet = open(simSetFile)
    senSum = 0
    wordTot = 0
    pairs_skipped = 0
    for line in simSet:
        split = line.split()
        w1 = split[0].lower()
        w2 = split[1].lower()
        out.write(w1 + ' ' + w2 + '\n')
        s = split[2]
        m1 = re.match('([a-z]+)\-[a-z]',w1)
        m2 = re.match('([a-z]+)\-[a-z]',w2)
        if m1: w1 = m1.group(1)
        if m2: w2 = m2.group(1)
        cAvgList = []
        cMaxList = []
        for vDict in vectorDicts:
            vecSim = avgSimPair(w1,w2,vDict,senSum,wordTot,out)
            if vecSim is None: 
                out.write('\n' + w1 + ' or ' + w2 + ' missing!\n')
                pairs_skipped += 1
                continue
            cAvgList.append(vecSim[0])
            cMaxList.append(vecSim[1])
        if mix in ('a','avg'):
            vecSimAvg = numpy.mean(cAvgList)
            vecSimMax = numpy.mean(cMaxList)
        elif mix in ('m','max'):
            vecSimAvg = max(cAvgList)
            vecSimMax = max(cMaxList)
        else: 
            print 'Specify mix type for combo!'
            sys.exit()
        
        senSum = vecSim[3]
        wordTot = vecSim[4]
        
        simSetSims.append(float(s))
        vecSimsAvg.append(vecSimAvg)
        vecSimsMax.append(vecSimMax)
        
        out.write(s + '\n')
        out.write(' '.join(str(e) for e in vecSim[:2]) + '\n')
        out.write('MAX: ' + ' '.join([str(e) for e in vecSim[2]])+ '\n\n')
        
        
    rho,p = scipy.stats.spearmanr(vecSimsAvg,simSetSims)
    rho2,p2 = scipy.stats.spearmanr(vecSimsMax,simSetSims)
    out.write('RHO (avg): ' + str(rho)+ '\n')
    out.write('RHO (max): ' + str(rho2)+ '\n')
    out.write('Avg # senses: ' + str(senSum/float(wordTot))+ '\n')
    out.write(str(pairs_skipped) + ' PAIRS SKIPPED FROM ' + setName+ '\n')
    out.close()
    return rho, rho2,p,pairs_skipped
    
def iterSimSets(vectorList, genFormatList, simSetFiles, mix):
    vecDictList = []
    for i in range(len(vectorList)):
        if type(vectorList[i]) == list:
            cVecDicts = []
            for cVec in vectorList[i]:
                cVecDicts.append(readVectors(cVec))
            break  
        if genFormatList[i] == 'gen': 
            vecDictList.append(loadVectors(vectorList[i]))
        elif genFormatList[i] == 'text': 
            vecDictList.append(readVectors(vectorList[i]))
        else:
            sys.stderr.write('Specify format: \'text\' or \'gen\'\n') 
            sys.exit()
    rhoListsAll = []
    for set in simSetFiles:
        m = re.match('.+/([^/]+)$',set)
        setName = m.group(1)
        rhoList = []
        for i in range(len(vectorList)):
            vectorFile = vectorList[i]
            if type(vectorFile) != list:
                vectorDict = vecDictList[i]
                mv = re.match('.+/([^/]+/[^/]+)$',vectorFile)
                vecName = mv.group(1) 
            else: vecName = 'COMBO-'+','.join(vectorFile)
            if type(vectorFile) == list:
                rho,rho2,p,skipped = getComboSpearman(cVecDicts,set,setName,vectorFile,mix)
            else:
                rho,rho2,p,skipped = getSpearman(vectorDict,set,setName,vectorFile)
            rhoList.append((setName,vecName,rho,rho2,str(skipped) + ' skipped'))
        rhoListsAll.append(rhoList)

    print 'SIM RESULTS'    
    for listA in rhoListsAll:
        for item in listA: print item
    print oov
    
        
if __name__ == "__main__":
    vecList,setList,genFormatList,mix = readCommandLineInput(sys.argv)
    iterSimSets(vecList,genFormatList,setList,mix)