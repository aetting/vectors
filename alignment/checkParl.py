#!/usr/bin/python
# -*- coding: utf8 -*-

import os, sys, re, itertools, numpy, math, scipy


class SenseObj(object):
    def __init__(self,lemma,sense):
        self.lemma = lemma
        self.sense = sense

def combineLayers(parldir,zh_annotdir,en_annotdir,mapdir):
    print 'combining corpus layers'
    parldir = os.path.abspath(parldir)
    zh_annotdir = os.path.abspath(zh_annotdir)
    en_annotdir = os.path.abspath(en_annotdir)
    mapdir = os.path.abspath(mapdir)
        
    enLemmaCounts = {}

    parlfiles = os.listdir(parldir)
    align_i = 0
    f_i = 0
    words = {}
    wordfreqs = {}
    inflections = {}
    
    ##iterate through parallel file pairs
    for f in parlfiles:
        f_i += 1
        parl_i = 0
        
        ##get parallel file IDs for this pair, and IDs of corresponding annotation files 
        m = re.match('([0-9]+_)(.+)\.tok\.ne.+',f)
        if not m: continue
        zhID = m.group(2)
        parlID = m.group(1) + m.group(2)
        trf = open(os.path.join(zh_annotdir,zhID + '.parallel'))
        trl = trf.read().split('\n')[1].split()[2]
        t = re.match('.+/([^/]+)',trl)
        enID = t.group(1)
        trf.close()
        
        ##create lists of sentences in two parallel files
        enParlFile = open(os.path.join(parldir,parlID+'.tok.train.declass'))
        zhParlFile = open(os.path.join(parldir,parlID+'.tok.ne.train.declass'))
        enLineList = enParlFile.read().split('\n')
        enLines = [e for e in enLineList if len(e) > 0]
        zhLineList = zhParlFile.read().split('\n')
        zhLines = [e for e in zhLineList if len(e) > 0]
        if len(enLines) != len(zhLines): print parlID + ': zh and en don\'t have same number of lines'
        
        ##create dict mapping sentence and word index in parallel file to (not yet corrected) positions in sense annotation (from parse trees)
        enMapFile = open(os.path.join(mapdir,'mapping_'+zhID+'.tok.train.declass'))
        enMapLines = [e for e in enMapFile.read().split('\n') if len(e) > 0]
        enMaps = {}
        for m in enMapLines:
            enMaps[m.split()[0]] = m.split()[1]
        enMapFile.close()
        
        ##create dict mapping sentence/word indices to sense annotations (entries only for positions that have English annotation)
        enSenseFile = open(os.path.join(en_annotdir,enID+'.sense'))
        enSenseLines = [e for e in enSenseFile.read().split('\n') if len(e) > 0]
        enSenseAnnotatedPos = {}
        
        for s in enSenseLines:
            pos = s.split()[1]+ '_' + s.split()[2]
            lemma = s.split()[3]
            if len(s.split()) == 5: sense = s.split()[4]
            else: sense = s.split()[5]
            thisSense = SenseObj(lemma,sense)
            enSenseAnnotatedPos[pos] = thisSense
            if not enLemmaCounts.has_key(lemma): enLemmaCounts[lemma] = 0
            enLemmaCounts[lemma] += 1
        enSenseFile.close()
        
    
        ##iterate through lines of parallel file pair
        parlAligns = {}
        for i in range(len(enLines)):
            enLine = enLines[i]
            zhLine = zhLines[i]
            
            enLineWords = enLine.split() 
            zhLineWords = zhLine.split() 

            
            ##iterate through words in line, locating corresponding word position used by sense annotation (using mapping fix if applicable)
            for wp in range(len(enLineWords)):
                enParlPos = str(parl_i) + '_' + str(wp)
                enOrigPos = enMaps[enParlPos]

                senPos = enOrigPos

                ##for positions with sense annotation, write to dict storing aligned word, sense annotation, and position in parallel file
                ##keys are LEMMAS, which is how the sense annotation is done
                if enSenseAnnotatedPos.has_key(senPos):
                    lem = enSenseAnnotatedPos[senPos].lemma
                    sense = enSenseAnnotatedPos[senPos].sense
                    enWord = enLineWords[wp].lower()
                    if lem[0] != enWord[0] and not enWord[0] == "'" and not re.match('(is|was|are|were|am|went)',enWord): 
                        print lem + ' ' + enWord

            parl_i += 1
            align_i += 1
        enParlFile.close()
        zhParlFile.close()

    
    return [words]
    
if __name__ == "__main__":
    combineLayers(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])