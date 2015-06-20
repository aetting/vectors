import os, sys, numpy
from matplotlib import pyplot

def plotFromOutput(syn,poly,freq):

    synfile = open(os.path.abspath(syn))
    polyfile = open(os.path.abspath(poly))
    freqfile = open(os.path.abspath(freq))
    
    syn_all = []
    poly_all = []
    freqdict = {}
    freqlist = []
    minZFreqlistP = []
    minZFreqlistS = []
    
    for line in freqfile:
        s = line.split(',')
        freqdict[s[0]] = float(s[1])
    
    for line in synfile:
        s = line.split(',')
        lem = s[0]
        zFreqs = [float(s[5]),float(s[6])]
        freqlist.append(freqdict[lem])
        minZFreqlistS.append(min(zFreqs))
        
    
    for line in polyfile:
        s = line.split(',')
        lem = s[0] 
        zFreqs = [float(s[5]),float(s[6])]
        freqlist.append(freqdict[lem])
        minZFreqlistP.append(min(zFreqs))  
             
#     freqarray = numpy.array(freqlist)
#     top = numpy.percentile(freqarray,100)
#     bot = numpy.percentile(freqarray,0)
    
    minZarrayS = numpy.array(minZFreqlistS)
    minZarrayP = numpy.array(minZFreqlistP)
    
    up = 100
    low = 90
    topS = numpy.percentile(minZarrayS,up)
    botS = numpy.percentile(minZarrayS,low)
    topP = numpy.percentile(minZarrayP,up)
    botP = numpy.percentile(minZarrayP,low)
    
    
    print topS
    print botS
    print topP
    print botP
    
    
    synfile.close()
    polyfile.close()
    freqfile.close()
    
    synfile = open(os.path.abspath(syn))
    polyfile = open(os.path.abspath(poly))
               
    for line in synfile:
        s = line.split(',')
        sim = float(s[3])
        lem = s[0]
        zFreqs = [float(s[5]),float(s[6])]
        minZ = min(zFreqs)
#         if bot <= freqdict[lem] <= top: syn_all.append(sim)
        if botS <= minZ <= topS: syn_all.append(sim)
#     print syn_all
    
    for line in polyfile:
        s = line.split(',')
        sim = float(s[3])
        lem = s[0]
        zFreqs = [float(s[5]),float(s[6])]
        minZ = min(zFreqs)
#         if bot <= freqdict[lem] <= top: poly_all.append(sim)
        if botP <= minZ <= topP: poly_all.append(sim) 
#     print poly_all

    synfile.close()
    polyfile.close()
    freqfile.close()
    
    syn_mean = numpy.mean(syn_all)
    poly_mean = numpy.mean(poly_all)
    simthresh = numpy.mean([poly_mean,syn_mean])
    
    synfile = open(os.path.abspath(syn))
    polyfile = open(os.path.abspath(poly))
    
    tot_ct = 0
    syn_ct = 0
    poly_ct = 0
    tot_corr = 0
    syn_corr = 0
    poly_corr = 0
    poly_all2 = []
    syn_all2 = []
    
    for line in synfile:
        s = line.split(',')
        lem = s[0]
        zFreqs = [float(s[5]),float(s[6])]
        minZ = min(zFreqs)
#         if not bot <= freqdict[lem] <= top: continue
        if not botS <= minZ <= topS: continue
        sim = float(s[3])
        label = s[4]
        tot_ct += 1
        syn_ct += 1
        syn_all2.append(sim)
        if sim >= simthresh: 
            tot_corr += 1
            syn_corr += 1
            
    for line in polyfile:
        s = line.split(',')
        lem = s[0]
        zFreqs = [float(s[5]),float(s[6])]
        minZ = min(zFreqs)
#         if not bot <= freqdict[lem] <= top: continue
        if not botP <= minZ <= topP: continue
        sim = float(s[3])
        label = s[4]
        tot_ct += 1
        poly_ct += 1
        poly_all2.append(sim)
        if sim < simthresh: 
            tot_corr += 1
            poly_corr += 1
            
    poly_acc = float(poly_corr)/poly_ct
    syn_acc = float(syn_corr)/syn_ct
    tot_acc = float(tot_corr)/tot_ct
    
    print 'syn acc: ' + str(syn_acc) + ', poly acc: ' + str(poly_acc) + ', total acc: ' + str(tot_acc)
            
    pyplot.hist(poly_all2,bins = 200,alpha=.5,label = 'poly')
    pyplot.hist(syn_all2,bins = 200,alpha=.5,label = 'syn')
    pyplot.legend(loc='upper right')
    pyplot.savefig('fullOntoDist_pfo.png')
    pyplot.clf()
    
    synfile.close()
    polyfile.close()
    
if __name__ == "__main__":
    plotFromOutput(sys.argv[1],sys.argv[2],sys.argv[3])    