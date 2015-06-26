#!/usr/bin/python
# -*- coding: utf8 -*-

import os, sys, re, itertools, gensim, numpy, math, scipy
from scipy import stats
from matplotlib import pyplot
from sys import stdout


class SenseObj(object):
    def __init__(self,lemma,sense):
        self.lemma = lemma
        self.sense = sense      
    
def runExperiment(parldir,zh_annotdir,en_annotdir,aligndir,mapdir,w2vmodel):
    
    [senseDict,counts] = combineLayers(parldir,zh_annotdir,en_annotdir,aligndir,mapdir)
    print 'loading model'
    vecmodel = gensim.models.Word2Vec.load(os.path.abspath(w2vmodel))
    print 'running experiment'
    pairs = []
    testi = 0
    pairs_added = 0
    for w,alignwdict in senseDict.items():
        if len(alignwdict) < 2: continue
        pivotlist = []
#         print w
        for alignw,tokList in alignwdict.items():
#             print alignw
            alignwsenses = {}
            for i in range(len(tokList)):
#                 print tokList[i]
                if not alignwsenses.has_key(tokList[i][0]): alignwsenses[tokList[i][0]] = 0
                alignwsenses[tokList[i][0]] += 1
            max = 0
            for s,n in alignwsenses.items():
                if n > max: 
                    max = n
                    sense = s
#             print sense 
            pivotlist.append([alignw,sense,w])
        pivotpairs = []
        pairIterator = itertools.combinations(pivotlist,2)
#         print list
        testi += 1
        for pair in pairIterator:
            pivotpairs.append(pair)
            pairs_added += 1
#             print ' '.join([pair[0][0],pair[0][1],pair[1][0],pair[1][1]])
        pairs += pivotpairs    
    
    test = []
    syn_sim_list = []
    syn_sim_list_ALL = []
    syn_tot = 0
    poly_sim_list = []
    poly_sim_list_ALL = []
    poly_tot = 0
    
    items_tot = 0
    
    synOut = open('syn_output.txt','w')
    polyOut = open('poly_output.txt','w')

    for pair in pairs:
        try: vecmodel.similarity(pair[0][0],pair[1][0])
        except: 
            continue
        else: 
            sim = vecmodel.similarity(pair[0][0],pair[1][0])
            items_tot += 1
            if pair[0][1] == pair[1][1]: 
                label = 'syn'
                syn_sim_list_ALL.append(sim)
                synOut.write(pair[0][2] + ',' + pair[0][0] + ',' + pair[1][0] + ',' + str(sim) + ',' + label + ',' + str(counts[pair[0][0]]) + ',' + str(counts[pair[1][0]]) + '\n')
                if items_tot % 10 == 0: test.append((sim, label, pair[0][0], pair[1][0], pair[0][2]))
                else: 
                    syn_sim_list.append(sim)
                    syn_tot += 1
            else: 
                label = 'poly'
                poly_sim_list_ALL.append(sim)
                polyOut.write(pair[0][2] + ',' + pair[0][0] + ',' + pair[1][0] + ',' + str(sim) + ',' + label + ',' + str(counts[pair[0][0]]) + ',' + str(counts[pair[1][0]]) + '\n')
                if items_tot % 10 == 0: test.append((sim, label, pair[0][0],pair[1][0], pair[0][2]))
                else: 
                    poly_sim_list.append(sim)
                    poly_tot += 1
    
    synOut.close()
    polyOut.close()
    print 'pairs added: ' + str(pairs_added)
                    
    print '\n'
    polymean = numpy.mean(poly_sim_list)
    print 'poly mean: ' + str(polymean)
    synmean = numpy.mean(syn_sim_list)
    print 'syn mean: ' + str(synmean)
    
    simthresh = numpy.mean([polymean,synmean])
    print 'threshold: ' + str(simthresh)
    test_tot = len(test)
    
    [t,p] = stats.ttest_ind(poly_sim_list_ALL,syn_sim_list_ALL)
    print '\n'
    print 'Full dataset t-test: '
    print 't: ' + str(round(t,3)) + ' p: ' + str(round(p,6)) 
    
    pyplot.hist(poly_sim_list_ALL,bins = 200,alpha=.5,label = 'poly')
    pyplot.hist(syn_sim_list_ALL,bins = 200,alpha=.5,label = 'syn')
    pyplot.legend(loc='upper right')
    pyplot.savefig('fullOntoDist.png')
    pyplot.clf()
            
    syn_correct = 0
    poly_correct = 0
    vec_correct = 0
    vec_syn_guesses = 0
    vec_poly_guesses = 0
    vec_poly_correct= 0
    vec_syn_correct= 0
    
    errorfile = open('errorPairs.txt','w')
    #run on held-out test set
    for sim,label,w1,w2,engword in test:
        simscore = float(sim)
        if simscore >= simthresh: 
            guess = 'syn'
            vec_syn_guesses += 1
        else: 
            guess = 'poly'
            vec_poly_guesses += 1
        if label == 'poly': 
            poly_correct += 1
            if guess == label: vec_poly_correct += 1
        elif label == 'syn': 
            syn_correct += 1
            if guess == label: vec_syn_correct += 1
        if guess == label: vec_correct += 1
        else: errorfile.write(engword + ' ' + w1 + ' ' + w2 + ' ' + str(round(float(sim),3)) + ' ' + label + '\n')
    errorfile.close()
    
    #precision = correct/guesses of that type
    #recall = correct/actual things of that type
    p_p_precision = 100*round(poly_correct/float(test_tot),3)
    p_p_recall = 100*round(poly_correct/float(poly_correct),3)
    
    s_s_precision = 100*round(syn_correct/float(test_tot),3)
    s_s_recall = 100*round(syn_correct/float(syn_correct),3)
    
    v_p_precision = 100*round(vec_poly_correct/float(vec_poly_guesses),3)
    v_p_recall = 100*round(vec_poly_correct/float(poly_correct),3)
    v_s_precision = 100*round(vec_syn_correct/float(vec_syn_guesses),3)
    v_s_recall = 100*round(vec_syn_correct/float(syn_correct),3)
    
    syn_accuracy = 100*round(syn_correct/float(test_tot),3)
    poly_accuracy = 100*round(poly_correct/float(test_tot),3)
    vec_accuracy = 100*round(vec_correct/float(test_tot),3)
    
    print '\n'
    print 'Polysemy-only baseline: '
    print 'Poly: precision = ' + str(p_p_precision) + ' recall = ' + str(p_p_recall)
    print 'Syn: precision = 0 recall = 0'
    print 'Total accuracy: ' + str(poly_accuracy)
    print '\n'
    print 'Synonymy-only baseline: '
    print 'Poly: precision = 0 recall = 0'
    print 'Syn: precision = ' + str(s_s_precision) + ' recall = ' + str(s_s_recall)
    print 'Total accuracy: ' + str(syn_accuracy)
    print '\n'
    print 'Vectors: '
    print 'Poly: precision = ' + str(v_p_precision) + ' recall = ' + str(v_p_recall)
    print 'Syn: precision = ' + str(v_s_precision) + ' recall = ' + str(v_s_recall) 
    print 'Total accuracy: ' + str(vec_accuracy)           



def combineLayers(parldir,zh_annotdir,en_annotdir,aligndir,mapdir):
# def combineLayers():
    print 'combining corpus layers'
    parldir = os.path.abspath(parldir)
    zh_annotdir = os.path.abspath(zh_annotdir)
    en_annotdir = os.path.abspath(en_annotdir)
    mapdir = os.path.abspath(mapdir)
#     mapfixdir = os.path.abspath(mapfixdir)

    alignDoc = open(os.path.join(os.path.abspath(aligndir),'training.align'))
    enAligndoc = open(os.path.join(os.path.abspath(aligndir),'training.tok.train.declass'))
    zhAligndoc = open(os.path.join(os.path.abspath(aligndir),'training.tok.ne.train.declass'))
    alignLines = alignDoc.read().split('\n')
    enAlignLines = enAligndoc.read().split('\n') 
    zhAlignLines = zhAligndoc.read().split('\n') 
    alignDoc.close()
    enAligndoc.close()
    zhAligndoc.close()
    
    #run cleanAlignments function and get alignments and numbers for the entire alignment corpus
    [translations,counts,num_alignments] = cleanAlignments(aligndir)
    
    enLemmaCounts = {}

#     parlfiles = os.listdir(parldir)
    parlfiles = ['0010_cnn_0003.tok.ne.train.declass','0011_cnn_0004.tok.ne.train.declass','0012_msnbc_0000.tok.ne.train.declass','0013_phoenix_0000.tok.ne.train.declass','0014_phoenix_0007.tok.ne.train.declass','0015_phoenix_0009.tok.ne.train.declass','0016_phoenix_0011.tok.ne.train.declass','001_cctv_0000.tok.ne.train.declass','002_cctv_0001.tok.ne.train.declass','003_cctv_0002.tok.ne.train.declass','004_cctv_0003.tok.ne.train.declass','005_cctv_0004.tok.ne.train.declass','006_cctv_0005.tok.ne.train.declass','007_cnn_0000.tok.ne.train.declass','008_cnn_0001.tok.ne.train.declass','009_cnn_0002.tok.ne.train.declass']
    align_i = 0
    lines_completed = 0
    words_added = 0
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
#         print f
        num=m.group(1)
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
        enMapFile = open(os.path.join(mapdir,'mapping_'+num+zhID+'.tok.train.declass'))
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
#             if f_i < 2: print pos + '...' + enSenseAnnotatedPos[pos].lemma + ',' + enSenseAnnotatedPos[pos].sense
        enSenseFile.close()
        
        ##create dict with entries for sentences that have length discrepancies between sense annotation and parallel files -- for fixing position mapping
#         enMapFixFile = open(os.path.join(mapfixdir,zhID+'_enmapfix.txt'))
#         enMapFixLines = [e for e in enMapFixFile.read().split('\n') if len(e) > 0]
#         enMapFixes = {}
#         for m in enMapFixLines:
#             enMapFixes[m.split()[0]] = 1
#         enMapFixFile.close()
    
        ##iterate through lines of parallel file pair
        parlAligns = {}
        for i in range(len(enLines)):
            enLine = enLines[i]
            zhLine = zhLines[i]
            
            ##create dict mapping English sentence/word index in parallel file to Chinese sentence/word index
            for a in alignLines[align_i].split():
                enPos = str(parl_i) + '_' + a.split('-')[1]
                zhPos = str(parl_i) + '_' + a.split('-')[0]
                parlAligns[enPos] = zhPos

#             enLineWords = [e for e in enLine.split() if not re.match('\(.*',e)]
#             zhLineWords = [e for e in zhLine.split() if not re.match('\(.*',e)]
            enLineWords = enLine.split() 
            zhLineWords = zhLine.split() 
            enAlLineWords = enAlignLines[align_i].split()
            zhAlLineWords = zhAlignLines[align_i].split()
            if len(enLineWords) != len(enAlLineWords) or len(zhLineWords) != len(zhAlLineWords):
#                 print 'length mismatch'
#                 print enLineWords
#                 print enAlLineWords
#                 print ' '.join(zhLineWords)
#                 print ' '.join(zhAlLineWords)
#                 print align_i
                align_i += 1
                parl_i += 1
                continue
            
            offsetLine = 0
            
            ##iterate through words in line, locating corresponding word position used by sense annotation (using mapping fix if applicable)
            skip = 0
            for wp in range(len(enLineWords)):
#                 if f_i < 2: print enLineWords
                enParlPos = str(parl_i) + '_' + str(wp)
#                 if not enMaps.has_key(enParlPos): break
                enOrigPos = enMaps[enParlPos]
#                 if enMapFixes.has_key(enOrigPos): offsetLine = 1
#                 if offsetLine == 1:
#                     newPos = int(enOrigPos.split('_')[1]) + 1
#                     senPos = enOrigPos.split('_')[0] + '_' + str(newPos)
#                 else: senPos = enOrigPos
                senPos = enOrigPos
                

                ##for positions with sense annotation, write to dict storing aligned word, sense annotation, and position in parallel file
                ##keys are LEMMAS, which is how the sense annotation is done
                if enSenseAnnotatedPos.has_key(senPos):
                    lem = enSenseAnnotatedPos[senPos].lemma
                    sense = enSenseAnnotatedPos[senPos].sense
                    if not parlAligns.has_key(enParlPos): continue
                    alignWordPos = int(parlAligns[enParlPos].split('_')[1])
                    if enLineWords[wp] != enAlLineWords[wp] or zhLineWords[alignWordPos] != zhAlLineWords[alignWordPos]:
#                         print 'word mismatch'
#                         print enLineWords
#                         print enAlLineWords
#                         print ' '.join(zhLineWords)
#                         print ' '.join(zhAlLineWords)
                        break
                    alignWord = zhLineWords[alignWordPos]
                    enWord = enLineWords[wp]
#                     enWord = enLineWords[wp].lower()
                    if lem[0] != enWord[0] and not enWord[0] == "'" and not re.match('(is|was|are|were|am|went)',enWord): 
                        print lem
                        print enWord
                        print enID
                        print i
                        print f
                        print align_i
                    if not inflections.has_key(lem): inflections[lem] = {}
                    if not inflections[lem].has_key(enWord): inflections[lem][enWord] = counts[enWord]
                    if not translations[enWord].has_key(alignWord): 
                        print 'trans key missing'
                        print enWord
                        print alignWord
                        print enLine
                        print enAlignLines[align_i]
                        print ' '.join(zhLineWords)
                        print ' '.join(zhAlLineWords)
                        print alignLines[align_i]
                    
                    #filter by PMI and filter out pronouns and auxiliary/light verbs
                    cE = counts[enWord]
                    cZ = counts[alignWord]
                    t = translations[enWord][alignWord]
                    pxy = t/float(num_alignments)
                    px = cE/float(num_alignments)
                    py = cZ/float(num_alignments)
                    pmi_frac = pxy/(px*py)
                    pmi = math.log(pmi_frac,2)
                    perc = t/float(cE)
                    b = re.match('(have|be|do|get)\-.*',lem)
                    p = re.match('(我们|你们|你|我|他|她|他们|您|您们)',alignWord)
                    if pmi < 8 or b or p: continue 
                    
                    if not words.has_key(lem): words[lem] = {}
                    if not words[lem].has_key(alignWord): words[lem][alignWord] = []
                    words[lem][alignWord].append([sense, enParlPos])
                    words_added += 1
                    if not wordfreqs.has_key(lem): wordfreqs[lem] = {}
                    if not wordfreqs[lem].has_key(alignWord): wordfreqs[lem][alignWord] = cZ
            parl_i += 1
            align_i += 1
            lines_completed += 1
        enParlFile.close()
        zhParlFile.close()
#     print words.items()[1]

    enLemmaCtFile = open('enLemmaCt.txt','w')
    for lem,ct in enLemmaCounts.items():
        enLemmaCtFile.write(lem+','+str(ct)+'\n')
    enLemmaCtFile.close()
    transFreqFile = open('transFreq.txt','w')
    for lem,transdict in wordfreqs.items():
        transFreqFile.write(lem+';')
        for trans,freq in transdict.items():
            transFreqFile.write(trans+':'+str(freq)+',')
        transFreqFile.write('\n')
    transFreqFile.close()
    inflectFile = open('inflections.txt','w')
    for lem,enWordDict in inflections.items():
        inflectFile.write(lem+'; ')
        totct = 0
        for inf,ct in enWordDict.items():
            totct += ct
            inflectFile.write(inf+': '+str(ct)+',')
        inflectFile.write(';'+str(totct))
        inflectFile.write('\n')
    inflectFile.close()
    print 'lines completed: ' + str(lines_completed)
    print 'all lines: ' + str(align_i)
    print 'words added: ' + str(words_added)
    
    return [words,counts]
    
def cleanAlignments(aligndir):

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
            if not counts.has_key(enWord): counts[enWord] = 0
            if not counts.has_key(zhWord): counts[zhWord] = 0
            counts[enWord] += 1
            counts[zhWord] += 1
            if not translations.has_key(enWord): translations[enWord] = {}
            if not translations[enWord].has_key(zhWord): translations[enWord][zhWord] = 0
            translations[enWord][zhWord] += 1
    print ''

    return [translations,counts,num_alignments]
    
if __name__ == "__main__":
    runExperiment(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])

# if __name__ == "__main__":
#     cleanAlignments('/Users/allysonettinger/Desktop/300k_align')
    
# if __name__ == "__main__":
#     combineLayers(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
#     combineLayers('parl','zh','en','berkeleyOutput','mappings','enmapfix')
