#!/usr/bin/python
# -*- coding: utf8 -*-

### python bclassifyAlignments.py 'newparl' 'zh' 'en' '/Users/allysonettinger/Desktop/alignments/300k_align' 'newmap2' '/Users/allysonettinger/Desktop/zhModels/zhModel3' 'en' 1

import os, sys, re, itertools, gensim, numpy, math, scipy, sklearn, operator
from scipy import stats
from matplotlib import pyplot
from numpy import linalg
from sys import stdout
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats.stats import pearsonr


class SenseObj(object):
    def __init__(self,lemma,sense):
        self.lemma = lemma
        self.sense = sense
        
def classify(parldir,zh_annotdir,en_annotdir,aligndir,mapdir,w2vmodel,pivotlang,typelvl):
    
    typelvl = int(typelvl)
    if typelvl: print 'TYPE = TRUE'
    else: print 'TYPE = FALSE'
    [X_train,y_train,X_test,y_test,X_train_items, X_test_items,lemma_div,training_length] = getPairs(parldir,zh_annotdir,en_annotdir,aligndir,mapdir,w2vmodel,pivotlang,typelvl)
    
    ker = 'rbf'
    clf = svm.SVC(kernel=ker) 
    print 'training SVM'
    clf.fit(X_train, y_train)

    
    ##make predictions and get evaluation metrics
    print 'testing'
    predictions = clf.predict(X_test)
    evals = precision_recall_fscore_support(y_test,predictions)
    supp = clf.support_
    decf = clf.decision_function(X_test)
    
    decf_sorted = numpy.argsort(decf)
    
    one_feature_vectors = []
    with open('model_analysis.txt','w') as modelAnalyze:
        modelAnalyze.write('FEATURE CORRELATIONS\n')
        for ftr in range(len(X_test[0])):
            modelAnalyze.write('Feature ' + str(ftr))
            one_feature = []
            for item in range(len(X_test)):
                one_feature.append(X_test[item][ftr])
            for item in range(len(X_train)):
                one_feature.append(X_train[item][ftr])
#             r = pearsonr(decf,one_feature)
#             modelAnalyze.write('\nCorr w/ decision func: ' + str(r) + '\n\n')
            one_feature_vectors.append(one_feature)
#         modelAnalyze.write('\n0-1 corr: ' + str(pearsonr(one_feature_vectors[0],one_feature_vectors[1]))+ '\n')
#         modelAnalyze.write('\n0-2 corr: ' + str(pearsonr(one_feature_vectors[0],one_feature_vectors[2]))+ '\n')
#         modelAnalyze.write('\n0-3 corr: ' + str(pearsonr(one_feature_vectors[0],one_feature_vectors[3]))+ '\n')
#         modelAnalyze.write('\n0-4 corr: ' + str(pearsonr(one_feature_vectors[0],one_feature_vectors[4]))+ '\n')
#         modelAnalyze.write('\n0-5 corr: ' + str(pearsonr(one_feature_vectors[0],one_feature_vectors[5]))+ '\n\n')
#         for f in one_feature_vectors: 
#             print max(f)
#             print min(f)
#             print numpy.mean(f)
#             print '\n'
        modelAnalyze.write('\nDECISION FUNCTION\n')
        for i in decf_sorted:
            modelAnalyze.write(str(decf[i]) + ': ')
            modelAnalyze.write(predictions[i] + ',')
            modelAnalyze.write(','.join(X_test_items[i]) + '\n')
        modelAnalyze.write('\n\nSUPPORT VECTORS\n')
        for s in supp:
            modelAnalyze.write(','.join(X_train_items[s]) + '\n')
        modelAnalyze.write('Training:' + str(len(X_train)) + '\n')
        modelAnalyze.write('Support vectors:' + str(len(supp))+ '\n')
    
    poly_prec = evals[0][0]
    poly_rec = evals[1][0]
    poly_fmeasure = evals[2][0]
    poly_true_tot = evals[3][0]
    
    syn_prec = evals[0][1]
    syn_rec = evals[1][1]
    syn_fmeasure = evals[2][1]
    syn_true_tot = evals[3][1]
    
    print '\n'
    print 'Kernel: ' + ker
    print 'Features: ' + str(len(X_train[0]))
    print 'Lemma div: ' + str(lemma_div)
    print 'Training set length: ' + str(training_length)
    print evals
          
    
def getPairs(parldir,zh_annotdir,en_annotdir,aligndir,mapdir,w2vmodel,pivotlang,typelvl):
    
    [senseDict,counts,inflectot,lem_translations,lem_translations_onto,training_length] = combineLayers(parldir,zh_annotdir,en_annotdir,aligndir,mapdir,pivotlang)
    print 'loading model'
    vecmodel = gensim.models.Word2Vec.load(os.path.abspath(w2vmodel))
    pairs = []
    transnums = {}
    entropies = {}
    testi = 0
    pairs_added = 0
    print 'getting double alignment pairs'
    tokeninfo = open('token-level_info.txt','w')
    sensesets = open('sense-sets.txt','w')
    for w,alignwdict in senseDict.items():
        if len(alignwdict) < 2: continue
        transnums[w] = len(alignwdict)
        pivotlist = []
        tokeninfo.write(w + '\n')
        lemsenses = {}
        for alignw,tokList in alignwdict.items():
            alignwsenses = {}
            senseratios = {}
            tokeninfo.write('--' + alignw + ': ' + str(len(tokList)) + '\n')
            for i in range(len(tokList)):
                if typelvl:
                    if not alignwsenses.has_key(tokList[i][0]): alignwsenses[tokList[i][0]] = 0
                    alignwsenses[tokList[i][0]] += 1
                else:
                    sense = tokList[i][0]
                    pivotlist.append([alignw,sense,w])
                if not lemsenses.has_key(tokList[i][0]): lemsenses[tokList[i][0]] = {}
                if not lemsenses[tokList[i][0]].has_key(alignw): lemsenses[tokList[i][0]][alignw] = 1
            if typelvl:
                mx = 0
                for s,n in alignwsenses.items():
                    if (float(n)/len(tokList)) < 1.0: tokeninfo.write(str(s) + ': ' + str(float(n)/len(tokList)) + '\n')
                    if n > mx: 
                        mx = n
                        sense = s
                pivotlist.append([alignw,sense,w])
        sensesets.write(w + '\n')
        for s,alignwdict in lemsenses.items():
            sensesets.write(s + '\n')
            for alignw,one in lemsenses[s].items():
                sensesets.write(alignw + '\n')
            
        pivotpairs = []
        pairIterator = itertools.combinations(pivotlist,2)
        testi += 1
        for pair in pairIterator:
            pivotpairs.append(pair)
            pairs_added += 1
        pairs += pivotpairs    
    tokeninfo.close()
    sensesets.close()
    
    
    lemma_dist = {}
    vocabmissing = {}
    total = 0
    
    test = []
    syn_sim_list = []
    syn_sim_list_ALL = []
    syn_tot = 0
    poly_sim_list = []
    poly_sim_list_ALL = []
    poly_tot = 0
    
    X_train = []
    X_test = []
    X_train_items = []
    X_test_items = []
    Y_train = []
    Y_test = []
    
    items_tot = 0
    
    synOut = open('syn_output.txt','w')
    polyOut = open('poly_output.txt','w')
    
    ##first pass through pairs to get feature spans
    allsims = []
    polysims = []
    synsims = []
    lt_ent_list = []
    lto_ent_list = []
    inflect_tot_list = []
    transn_list = []
    transnum_all_list = []
    w1freq_list = []
    w2freq_list = []
    zh_lowfreq_list = []
    zh_highfreq_list = []
    zhratio_list = []
    
    for pair in pairs:
        lem = pair[0][2]
        w1 = pair[0][0]
        w2 = pair[1][0]
        
        ##extract two types of entropy (totals based on Ontonotes stuff, versus totals based on whole alignment training set)        
        if not entropies.has_key(lem): 
            LT_total = 0
            LT_entropy = 0
            LTO_total = 0
            LTO_entropy = 0 
            for alignw,ct in lem_translations[lem].items():
                LT_total += ct
            for alignw,ct in lem_translations[lem].items():
                prob = float(ct)/LT_total
                ent = prob*math.log(prob,2)
                LT_entropy += ent
            for alignw,ct in lem_translations_onto[lem].items():
                LTO_total += ct
            for alignw,ct in lem_translations_onto[lem].items():
                prob = float(ct)/LTO_total
                ent = prob*math.log(prob,2)
                LTO_entropy += ent
            LT_entropy = -1*LT_entropy
            LTO_entropy = -1*LTO_entropy
            entropies[lem] = (LT_entropy,LTO_entropy)
            lt_ent_list.append(LT_entropy)
            lto_ent_list.append(LTO_entropy)
            
        w1freq = math.log(1+counts[w1],2)
        w2freq = math.log(1+counts[w2],2)
        zh_lowfreq = min([w1freq,w2freq])
        zh_highfreq = max([w1freq,w2freq])
        
        inflect_tot_list.append(math.log(1+inflectot[lem],2))
        transn_list.append(math.log(1+transnums[lem],2))
        transnum_all_list.append(math.log(1+len(lem_translations[lem]),2))
        w1freq_list.append(math.log(1+counts[w1],2))
        w2freq_list.append(math.log(1+counts[w2],2))
        zh_lowfreq_list.append(zh_lowfreq)
        zh_highfreq_list.append(zh_highfreq)
        zhratio_list.append(zh_lowfreq - float(zh_highfreq))
        
        if not lemma_dist.has_key(lem): lemma_dist[lem] = 0
        lemma_dist[lem] += 1
        total += 1 
          
        try: vecmodel.similarity(pair[0][0],pair[1][0])
        except: 
            continue
        else: 
            sim = vecmodel.similarity(pair[0][0],pair[1][0])
            allsims.append(sim)
            if pair[0][1] == pair[1][1]: synsims.append(sim)
            else: polysims.append(sim)
    
    
    meanSim = numpy.mean(allsims)
    polyMean = numpy.mean(polysims)
    synMean = numpy.mean(synsims)
    neutralMean = numpy.mean([polyMean,synMean])
    print 'NEUTRAL MEAN: ' + str(neutralMean) + '\n'
    
    sortedpairs = sorted(lemma_dist.items(), key=operator.itemgetter(1))
    s_i = 0
    test_tot = 0
    lemma_set_assignments = {}
    if typelvl: lemma_div = 4
    else: lemma_div = 4
    for lem,ct in sortedpairs:
        s_i += 1
        if s_i % lemma_div == 0 and test_tot < (float(total)/10):
            lemma_set_assignments[lem] = 'test'
            test_tot += ct
        else: lemma_set_assignments[lem] = 'train'
    print 'test set total: ' + str(test_tot)     
    with open('lemmadist.txt','w') as lemmadistfile:
        for lem,set in lemma_set_assignments.items():
            if set == 'test': lemmadistfile.write(lem + ': ' + str(lemma_dist[lem]) + ': ' + set + '\n')
        for lem,set in lemma_set_assignments.items():
            if set == 'train': lemmadistfile.write(lem + ': ' + str(lemma_dist[lem]) + ': ' + set + '\n')
    
    print 'Total in lemma dist file: ' + str(total)
    
    ##iterate through all pairs and assign feature vectors
    for pair in pairs:
        try: vecmodel.similarity(pair[0][0],pair[1][0])
        except: 
#             sim = neutralMean
            sim = meanSim
#             sim = 'missing'
        else: 
            sim = vecmodel.similarity(pair[0][0],pair[1][0])
#         if sim == 'missing': continue
        items_tot += 1
        lem = pair[0][2]
        w1 = pair[0][0]
        w2 = pair[1][0]
        if pair[0][1] == pair[1][1]: 
            label = 'syn'
            syn_sim_list_ALL.append(sim)
            synOut.write(','.join([lem,w1, w2,str(sim),label,str(counts[w1]), str(counts[w2])]) + '\n')
            if items_tot % 10 == 0: 
                test.append((sim, label, w1, w2, lem))
            else: 
                syn_sim_list.append(sim)
                syn_tot += 1
        else: 
            label = 'poly'
            poly_sim_list_ALL.append(sim)
            polyOut.write(','.join([lem, w1, w2, str(sim), label, str(counts[w1]), str(counts[w2])]) + '\n')
            if items_tot % 10 == 0: 
                test.append((sim, label, w1,w2, lem))
            else: 
                poly_sim_list.append(sim)
                poly_tot += 1

        ##extract dimensionwise similarity feature 
#         dimensions = []
#         norm_A = numpy.linalg.norm(vecmodel[w1])
#         norm_B = numpy.linalg.norm(vecmodel[w2])
#         for i in range(len(vecmodel[w1])):
#             a = vecmodel[w1][i]
#             b = vecmodel[w2][i]
#             dim = (a*b)/(norm_A*norm_B)
#             dimensions.append(dim)

        w1f = math.log(1+counts[w1],2)
        w2f = math.log(1+counts[w2],2)
        zh_lowf = min([w1f,w2f])
        zh_highf = max([w1f,w2f])
        
        inflect_tot = centerScale(math.log(1+inflectot[lem],2),inflect_tot_list)
        transn = centerScale(math.log(1+transnums[lem],2),transn_list)
        transnum_all = centerScale(math.log(1+len(lem_translations[lem]),2),transnum_all_list)
        w1freq = centerScale(math.log(1+counts[w1],2),w1freq_list)
        w2freq = centerScale(math.log(1+counts[w2],2),w2freq_list)
        zh_lowfreq = centerScale(min([w1f,w2f]),zh_lowfreq_list)
        zh_highfreq = centerScale(max([w1f,w2f]),zh_highfreq_list)
        zhratio = centerScale(zh_lowf - float(zh_highf),zhratio_list)
        ent_lt = centerScale(entropies[lem][0],lt_ent_list)
        ent_lto = centerScale(entropies[lem][1],lto_ent_list)

#         inflect_tot = math.log(1+inflectot[lem],2)
#         transn = math.log(1+transnums[lem],2)
#         transnum_all = math.log(1+len(lem_translations[lem]),2)
#         w1freq = math.log(1+counts[w1],2)
#         w2freq = math.log(1+counts[w2],2)
#         zh_lowfreq = min([w1freq,w2freq])
#         zh_highfreq = max([w1freq,w2freq])
#         zhratio = zh_lowfreq - float(zh_highfreq)
#         ent_lt = entropies[lem][0]
#         ent_lto = entropies[lem][1]
                                
        ##inflect_tot is a sum, for a given lemma, over the counts of each of its inflections -- counts based on full alignment training bitext, and inflection list based on inflections of lemma found in Ontonotes bitext (only place we have lemma annotation)
        ##transn is based on number of word types that this lemma is aligned to in Ontonotes, post filtering
        ##transnum_all should be the number of word types that this lemma is aligned to in the entire alignment training bitext, post filtering
        ##entropies[lem][0] is the entropy over all alignments found in full alignment training bitext, for inflections of lemma (after PMI filtering). this is based on "translations" dictionary collected while going through training bitext.
        ##entropies[lem][1] is the entropy over all alignments found in Ontonotes bitext for lemma (lemma annotation in Ontonotes means don't have to go via inflection dictionary for this entropy. but it's probably noisier, being a smaller sample size.)   
            
            
#         featureset = [sim]
#         featureset = [sim,inflect_tot,transn,transnum_all,ent_lt,ent_lto,w1freq,w2freq,zh_lowfreq,zh_highfreq,zhratio]
        featureset = [sim,inflect_tot,transn,transnum_all,ent_lt,ent_lto]
#         featureset += dimensions
            
        itemset = [lem,w1,w2,str(sim),str(label)]
        ##create input for SVM. every tenth item, or divide by lemmas      
#         if items_tot % 10 == 0:
        if lemma_set_assignments[lem] == 'test':
            X_test.append(featureset)
            Y_test.append(label)
            X_test_items.append(itemset)
        else:
            X_train.append(featureset)
            Y_train.append(label)
            X_train_items.append(itemset)
    
    synOut.close()
    polyOut.close()
    
    print 'pairs added pre word2vec filtering: ' + str(pairs_added)
    print 'pairs added after word2vec filtering: ' + str(items_tot)
                    
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
    print '\n' 
    
    return [X_train,Y_train,X_test,Y_test,X_train_items, X_test_items,lemma_div,training_length]         



def combineLayers(parldir,zh_annotdir,en_annotdir,aligndir,mapdir,pivotlang):
# def combineLayers():
    print 'combining corpus layers'
    parldir = os.path.abspath(parldir)
    zh_annotdir = os.path.abspath(zh_annotdir)
    en_annotdir = os.path.abspath(en_annotdir)
    mapdir = os.path.abspath(mapdir)
#     mapfixdir = os.path.abspath(mapfixdir)

    if pivotlang == 'en':
        pivotcode = 'tok.train'
        paircode = 'tok.ne.train'
    else:
        pivotcode = 'tok.ne.train'
        paircode = 'tok.train'
        
    alignDoc = open(os.path.join(os.path.abspath(aligndir),'training.align'))
    
    ##here it starts being the case that in most places "en---" means "pivotlang" and "zh---" means "pairlang" since originally I wrote this with no flexibility to switch languages and English was the pivot language
    enAligndoc = open(os.path.join(os.path.abspath(aligndir),'training.'+pivotcode+'.declass'))
    zhAligndoc = open(os.path.join(os.path.abspath(aligndir),'training.'+paircode+'.declass'))
    alignLines = alignDoc.read().split('\n')
    enAlignLines = enAligndoc.read().split('\n') 
    zhAlignLines = zhAligndoc.read().split('\n') 
    alignDoc.close()
    enAligndoc.close()
    zhAligndoc.close()
    
    #run cleanAlignments function and get alignments and numbers for the entire alignment corpus
    [translations,counts,num_alignments,training_length] = cleanAlignments(aligndir,pivotlang)
    
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
    lem_translations = {}
    lem_translations_onto = {}
    
    ##iterate through parallel file pairs
    for f in parlfiles:
        f_i += 1
        parl_i = 0
        
        ##get parallel file IDs for this pair, and IDs of corresponding annotation files 
        m = re.match('([0-9]+_)(.+)\.tok\.ne+',f)
        if not m: continue
#         print f
        num=m.group(1)
        
        ##this "zhID" really means "Chinese", though, since the parl filenames were based on the Chinese and not the English. likewise, enID really means English
        zhID = m.group(2)
        parlID = m.group(1) + m.group(2)
        trf = open(os.path.join(zh_annotdir,zhID + '.parallel'))
        trl = trf.read().split('\n')[1].split()[2]
        t = re.match('.+/([^/]+)',trl)
        enID = t.group(1)
        trf.close()

        if pivotlang == 'en':
            pivotID = enID
            pairID = zhID
            pivotdir = en_annotdir
        else:
            pivotID = zhID
            pairID = enID
            pivotdir = zh_annotdir
        
        ##create lists of sentences in two parallel files
        enParlFile = open(os.path.join(parldir,parlID+'.'+pivotcode+'.declass'))
        zhParlFile = open(os.path.join(parldir,parlID+'.'+paircode+'.declass'))
        enLineList = enParlFile.read().split('\n')
        enLines = [e for e in enLineList if len(e) > 0]
        zhLineList = zhParlFile.read().split('\n')
        zhLines = [e for e in zhLineList if len(e) > 0]
        if len(enLines) != len(zhLines): print parlID + ': zh and en don\'t have same number of lines'

        
        ##create dict mapping sentence and word index in parallel file to (not yet corrected) positions in sense annotation (from parse trees)
        enMapFile = open(os.path.join(mapdir,'mapping_'+num+zhID+'.'+pivotcode+'.declass'))
        enMapLines = [e for e in enMapFile.read().split('\n') if len(e) > 0]
        enMaps = {}
        for m in enMapLines:
            enMaps[m.split()[0]] = m.split()[1]
        enMapFile.close()
        
        ##create dict mapping sentence/word indices to sense annotations (entries only for positions that have English annotation)
        enSenseFile = open(os.path.join(pivotdir,pivotID+'.sense'))
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
            ##here, unlike elsewhere, zh and en actually are fixed to mean English and Chinese, since the positioning in the alignment file is fixed (Chinese first, English second)
            for a in alignLines[align_i].split():
                enPos = str(parl_i) + '_' + a.split('-')[1]
                zhPos = str(parl_i) + '_' + a.split('-')[0]
                if pivotlang == 'en': parlAligns[enPos] = zhPos
                else: parlAligns[zhPos] = enPos

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
#                         print enID
#                         print i
#                         print f
#                         print align_i
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
                    
                    if not lem_translations_onto.has_key(lem): lem_translations_onto[lem] = {}
                    if not lem_translations_onto[lem].has_key(alignWord): lem_translations_onto[lem][alignWord] = 0
                    lem_translations_onto[lem][alignWord] += 1
                    
                    if not lem_translations.has_key(lem): 
                        lem_translations[lem] = {}
                        for inf,ct in inflections[lem].items():
                            for alignw,ct2 in translations[inf].items():
                                cE = counts[inf]
                                cZ = counts[alignw]
                                t = translations[inf][alignw]
                                pxy = t/float(num_alignments)
                                px = cE/float(num_alignments)
                                py = cZ/float(num_alignments)
                                pmi_frac = pxy/(px*py)
                                pmi = math.log(pmi_frac,2)
                                if pmi >= 8:
                                    if not lem_translations[lem].has_key(alignw): lem_translations[lem][alignw] = 0
                                    lem_translations[lem][alignw] += t
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
    inflectot = {}
    for lem,enWordDict in inflections.items():
        inflectFile.write(lem+'; ')
        totct = 0
        for inf,ct in enWordDict.items():
            totct += ct
            inflectFile.write(inf+': '+str(ct)+',')
        inflectFile.write(';'+str(totct))
        inflectot[lem] = totct
        inflectFile.write('\n')
    inflectFile.close()
    
#     for lem,dict in lem_translations.items():
#         print 'LEMTRANS: ' + lem
#         for alignw,ct in dict.items():
#             print str(alignw) + ' ' + str(ct)
#         print 'LEMTRANSONTO: ' + lem
#         for alignw,ct in lem_translations_onto[lem].items():
#             print str(alignw) + ' ' + str(ct)
    print 'lines completed: ' + str(lines_completed)
    print 'all lines: ' + str(align_i)
    print 'words added: ' + str(words_added)
    
    return [words,counts,inflectot,lem_translations,lem_translations_onto,training_length]
    
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
    
def centerScale(val,vec):
    span = max(vec)-min(vec)
    newval = (val-min(vec))/float(span)
    return newval
    
if __name__ == "__main__":
    classify(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])

# if __name__ == "__main__":
#     cleanAlignments('/Users/allysonettinger/Desktop/300k_align')
    
# if __name__ == "__main__":
#     combineLayers(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
#     combineLayers('parl','zh','en','berkeleyOutput','mappings','enmapfix')
