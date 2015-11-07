#!/usr/bin/python
# -*- coding: utf8 -*-

import math,sys,re,os,getopt
from sys import stdout

def readLemmatizer():
    d = {}
    f = open('morph_english.flat')
    for line in f:
        if line.startswith(';;;'): continue
        list = line.split()
        d[list[0]] = list[1]
    return d
    
def readCommandLineInput(argv):
    try:
        try:
            #specify the possible option switches
            opts, _ = getopt.getopt(sys.argv[1:], "a:p:o:s:h:c:t:k:m:l:g:", ["aligndir=", "pivotlang=", "ontdir=", "stat=",
                                                              "statthresh=", "counthresh=", "top=", "k=", "mid=", "lemmatize=","logistic="])
        except: print 'INPUT INCORRECT'
        aligndir = None
        pivotlang = None
        ontdir = None
        statThresh = None
        countThresh = 5
        top=None
        k = None
        mid = None
        lemmatize = None
        logistic = None
        stat = None
        # option processing
        for option, value in opts:
            if option in ("-a", "--aligndir"):
                aligndir = value
            elif option in ("-p", "--pivotlang"):
                pivotlang = value
            elif option in ("-o", "--ontdir"):
                ontdir = value
            elif option in ("-s", "--stat"):
                stat = value
            elif option in ("-h", "--statthresh"):
                statThresh = value
            elif option in ("-c", "--counthresh"):
                countThresh = value
            elif option in ("-t", "--top"):
                top = value
            elif option in ("-k", "--k"):
                k = value
            elif option in ("-m", "--mid"):
                mid = value
            elif option in ("-l", "--lemmatize"):
                lemmatize = bool(int(value))
            elif option in ("-g", "--logistic"):
                logistic = bool(int(value))
            else:
                print "Doesn't match any option"
        return (aligndir,pivotlang,ontdir,stat,statThresh,countThresh,top,k,mid,lemmatize,logistic)
    except: print "Something else went wrong??"
        

def cleanAlignments(aligndir, pivotlang,lemmatize):
    
    ##first line to remove if not lemmatizing
    if lemmatize:
        lemmaDict = readLemmatizer()
    
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
            
            ##second line to remove if not lemmatizing
            if lemmatize:
                if enWord in lemmaDict: enWord = lemmaDict[enWord]
            
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
    
def filterOntology(translations,counts,num_alignments,stat,statThresh,countThresh,top,k,mid,logistic):
    #filter by PMI and filter out pronouns and auxiliary/light verbs
    #we want to use the logistic function for the weights that will determine how much we want to move toward vectors in the cluster
    #but first we need to decide which alignments to keep in identifying senses
    senseWgt = 1.
    parList = [stat,statThresh,top,k,mid]
    ontName = 'ontology-' + '-'.join(parList)
    num_alignments = float(num_alignments)
    statThresh = float(statThresh)
    countThresh = float(countThresh)
    top = float(top)
    k = float(k)
    mid = float(mid)
    ontologyDict = {}
    if stat == 'P':
        for pivotword,alignw,pmi in getPMI(translations,counts,num_alignments,statThresh,countThresh):
            if logistic: w = logisticFunction(pmi,top,k,mid)
            else: 
                w = senseWgt
            if not pivotword in ontologyDict: ontologyDict[pivotword] = {}
            ontologyDict[pivotword][alignw] = w
            print pmi
            print w
        print 'GOT ONTOLOGY FROM PMI'
    elif stat == 'G':
        for pivotword,alignw,gVal in getG(translations,counts,num_alignments,statThresh):
            if logistic: w = logisticFunction(gVal,top,k,mid) 
            else: 
                w = senseWgt          
            if not pivotword in ontologyDict: ontologyDict[pivotword] = {}
            ontologyDict[pivotword][alignw] = w
            print gVal
            print w
        print 'GOT ONTOLOGY FROM G'
    
    return ontologyDict,ontName

        
def getPMI(translations,counts,num_alignments,pmiThresh,countThresh):
    print 'GETTING PMI'
    pmiThresh = float(pmiThresh)
    countThresh = float(countThresh)       
    for pivotword,alignwordsdict in translations.items():
        print '\n'+ pivotword
        for alignw, t in alignwordsdict.items():
            print alignw
            cP = counts[pivotword]
            cA = counts[alignw]
            t = translations[pivotword][alignw]
            pxy = t/float(num_alignments)
            px = cP/float(num_alignments)
            py = cA/float(num_alignments)
            pmi_frac = pxy/(px*py)
            pmi = math.log(pmi_frac,2)
            perc = t/float(cP)

            if pmi < pmiThresh or t < countThresh: continue
            yield (pivotword,alignw,pmi)

    
def getG(translations,counts,num_alignments,gThresh):
    gThresh = float(gThresh)
    print 'GETTING G TEST'
    for pivotword,alignwordsdict in translations.items():
        print '\n'+ pivotword
        for alignw, t in alignwordsdict.items():
            print alignw
            cP = float(counts[pivotword])
            notP = num_alignments - cP
            cA = float(counts[alignw])
            notA = num_alignments - cA
            
            joint = float(translations[pivotword][alignw])
            p_notA = cP - joint
            a_notP = cA - joint
            neither = num_alignments - (joint + p_notA + a_notP)
            
            joint_Exp = (cP/num_alignments) * cA
            p_notA_Exp = (cP/num_alignments) * notA
            a_notP_Exp = (notP/num_alignments) * cA
            neither_Exp = (notP/num_alignments) * notA
            
            O = [joint,p_notA,a_notP,neither]
            E = [joint_Exp,p_notA_Exp,a_notP_Exp,neither_Exp]
            
            gSum = 0
            for i in range(len(O)):
                if O[i] == 0  or E[i] == 0: continue
                term = O[i]*math.log(O[i]/E[i])
                gSum += term
            gVal = 2*gSum
            
            if gVal < gThresh: continue
            
            yield (pivotword,alignw,gVal)
    
    
def printOntology(ontologyDict,ontdir,ontName):
    senseagWgt = 1.
    ontname = os.path.join(ontdir,ontName)
    with open(ontname,'w') as ontolDoc:
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
    
if __name__ == "__main__":
    (aligndir,pivotlang,ontdir,stat,statThresh,countThresh,top,k,mid,lemmatize,logistic) = readCommandLineInput(sys.argv) 
    [translations,counts,num_alignments,training_length] = cleanAlignments(aligndir, pivotlang,lemmatize)
    ontologyDict,ontName = filterOntology(translations,counts,num_alignments,stat,statThresh,countThresh,top,k,mid,logistic)
    printOntology(ontologyDict,ontdir,ontName)