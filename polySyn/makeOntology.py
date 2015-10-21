#!/usr/bin/python
# -*- coding: utf8 -*-

import math,sys,re,os
from sys import stdout

def cleanAlignments(aligndir, pivotlang):
    print 'getting alignment counts'
    alignDoc = open(os.path.join(os.path.abspath(aligndir),'training.align'))
    enAligndoc = open(os.path.join(os.path.abspath(aligndir),'training.tok.train.declass'))
    zhAligndoc = open(os.path.join(os.path.abspath(aligndir),'training.tok.ne.train.declass'))
    alignLines = alignDoc.read().split('\n')
    enAlignLines = enAligndoc.read().split('\n') 
    zhAlignLines = zhAligndoc.read().split('\n') 
    alignDoc.close()
    enAligndoc.close()
    zhAligndoc.close()
         
    translations = {}
    counts = {}
    num_alignments = 0
    training_length = len(alignLines)
    for i in range(len(alignLines)):
        if i % 5000 == 0:
            stdout.write('.')
            stdout.flush()
        alignLine = alignLines[i].split()
        zhLine = zhAlignLines[i].split()
        enLine = enAlignLines[i].split()

        for j in range(len(alignLine)):
            zhPos = int(alignLine[j].split('-')[0])
            enPos = int(alignLine[j].split('-')[1])
            if zhPos >= len(zhLine) or enPos >= len(enLine): 
#                 print zhLine
#                 print enLine
#                 print alignLine
                break 
            num_alignments += 1	
            zhWord = zhLine[zhPos]
            enWord = enLine[enPos]
            
            enWord = re.sub('[#|%]','-',enWord)
            zhWord = re.sub('[#|%]','-',zhWord)
            
            if not counts.has_key(enWord): counts[enWord] = 0
            if not counts.has_key(zhWord): counts[zhWord] = 0
            counts[enWord] += 1
            counts[zhWord] += 1
            
            if pivotlang == 'en': 
                pivotWord = enWord
                pairWord = zhWord
            else:
                pivotWord = zhWord
                pairWord = enWord
                
            if not translations.has_key(pivotWord): translations[pivotWord] = {}
            if not translations[pivotWord].has_key(pairWord): translations[pivotWord][pairWord] = 0
            translations[pivotWord][pairWord] += 1
    print ''
    
    return [translations,counts,num_alignments,training_length]
    
def filterOntology(translations,counts,num_alignments):
    #filter by PMI and filter out pronouns and auxiliary/light verbs
    #we want to use the logistic function for the weights that will determine how much we want to move toward vectors in the cluster
    #but first we need to decide which alignments to keep in identifying senses
    ontologyDict = {}
    k = 1
    top = 1
    mid = 8
    for pivotword,alignwordsdict in translations.items():
        print '\n'+ pivotword
        for alignw, t in alignwordsdict.items():
            cP = counts[pivotword]
            cA = counts[alignw]
            t = translations[pivotword][alignw]
            pxy = t/float(num_alignments)
            px = cP/float(num_alignments)
            py = cA/float(num_alignments)
            pmi_frac = pxy/(px*py)
            pmi = math.log(pmi_frac,2)
            perc = t/float(cP)
            if pmi < 8 or t < 3: continue
            print alignw
            print str(pmi) 
            if not pivotword in ontologyDict: ontologyDict[pivotword] = {}
            w = logisticFunction(pmi,top,k,mid)
            ontologyDict[pivotword][alignw] = w
            print w
    return ontologyDict
    
def printOntology(ontologyDict):
    senseagWgt = 1.
    with open('ontology','w') as ontolDoc:
        for pivotword,alignwordsdict in ontologyDict.items():
            for alignw, alignwWgt in alignwordsdict.items(): 
                otherWords = [a for a in alignwordsdict.items() if a[0] != alignw]
                ontolDoc.write(alignw + '%' + pivotword + '#' + str(senseagWgt)+ ' ')
                for word,alignWgt in otherWords:
                    ontolDoc.write(word + '%' + pivotword + '#' + str(alignWgt) + ' ')
                ontolDoc.write('\n')
                
def logisticFunction(x,top,k,mid):
    y = top/(1+math.exp(-k*(x-mid)))
    return y
    
                    
def compileOntology(aligndir,pivotlang):
    [translations,counts,num_alignments,training_length] = cleanAlignments(aligndir, pivotlang)
    ontologyDict = filterOntology(translations,counts,num_alignments)
    printOntology(ontologyDict)
    
if __name__ == "__main__":
    compileOntology(sys.argv[1],sys.argv[2])