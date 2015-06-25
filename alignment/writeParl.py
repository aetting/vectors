import re, os, sys

#written to be run one directory above the en and zh directories

def writeParallelFiles(zhDir,enDir,zhSuf,enSuf):
    i = 0
    for f in os.listdir(zhDir):
        m = re.match('(.*)\.parallel',f)
        if not m: continue
        file = open(os.path.abspath(zhDir + '/' + f)) 
        this = file.readline()
        par = file.readline()
        map = re.match('.+\/([^\/]+)',par.split()[2])
        parfile = map.group(1)
        presfile = m.group(1)
        if this.startswith('translated'):
            transdoc = presfile
            origdoc = parfile
            transdir = zhDir
            origdir = enDir
            t = 1
        elif this.startswith('original'):
            origdoc = presfile
            transdoc = parfile
            transdir = enDir
            origdir = zhDir
            t = 0
        file.close()
        i += 1
#         print this
#         print presfile
#         print parfile
#         print origdoc
#         print transdoc
#         print '\n'
        zhSent = extractSentFromParse(os.path.abspath(zhDir + '/' + presfile + '.parse'))
        enSent = extractSentFromParse(os.path.abspath(enDir + '/' + parfile + '.parse'))
#         if i == 1: 
#             print enSent[213]
#             for w in zhSent[283]: print w
#                 for w in s: print w
            
        if t == 0:
            trSent = enSent
            trSuf = enSuf
            orSent = zhSent
            orSuf = zhSuf
        elif t == 1:
            trSent = zhSent
            trSuf = zhSuf
            orSent = enSent
            orSuf = enSuf
        
        orOut = open('00' + str(i) + '_' + presfile + orSuf,'w')
        trOut = open('00' + str(i) + '_' + presfile + trSuf,'w')
        orMapOut = open('mapping'+ '_' + '00' + str(i) + '_' + presfile + orSuf,'w')
        trMapOut = open('mapping'+ '_' + '00' + str(i) + '_' + presfile + trSuf,'w')
            
        aFile = open(os.path.abspath(transdir + '/' + transdoc + '.parallel'))
        orLine = ''
        trLine = ''
        orMapPrev = -1
        trMapPrev = -1
        parLine = 0
        for line in aFile:
            if not line.startswith('map'): continue
            s = line.split()
            orMap = int(s[1])
            trMap = int(s[2])
#             if i == 1: 
#                 print s
#                 print orMapPrev
#                 print orMap
#                 print trMapPrev
#                 print trMap
            if orMap == orMapPrev:
                trLine = trLine + ' ' + ' '.join(trSent[trMap])
                for spT in range(len(trSent[trMap])):
                    ppT = spT + 1 + ppTHeld
                    trMapOut.write(str(parLine) + '_' + str(ppT) + ' ' + str(trMap) + '_' + str(spT) + '\n')
                ppTHeld = ppT
#                 if i == 1: 
#                     print 'ONE'
#                     print trLine
#                     print orLine
            elif trMap == trMapPrev:
                orLine = orLine + ' ' + ' '.join(orSent[orMap])
                for spO in range(len(orSent[orMap])):
                    ppO = spO + 1 + ppOHeld
                    orMapOut.write(str(parLine) + '_' + str(ppO) + ' ' + str(orMap) + '_' + str(spO) + '\n')
                ppOHeld = ppO
#                 if i == 1: 
#                     print 'TWO'
#                     print trLine
#                     print orLine
            elif orMap != orMapPrev and trMap != trMapPrev:
#                 if len(orLine) > 0 and len(trLine) > 0 and i == 1:
#                     print orLine
#                     print trLine
                if len(orLine) > 0 and len(trLine) > 0:
                    orOut.write(orLine + '\n')
                    trOut.write(trLine + '\n')
                    parLine += 1
                orLine = ' '.join(orSent[orMap])
                trLine = ' '.join(trSent[trMap])
                #****here need to loop through each word in orSent[orMap] (the sentence being written this line)
                #and record the parl line number that we're writing and the word position on that line, and link it to the sentence number
                #(orMap) and word position in the sentence (i in the loop through the words)
                # to get word position in parline, at end of loop through sentence words, keep last i, and if you later add to the line, 
                #add the kept i to each new i 
                for spO in range(len(orSent[orMap])):
                    ppO = spO
                    orMapOut.write(str(parLine) + '_' + str(ppO) + ' ' + str(orMap) + '_' + str(spO) + '\n')
                ppOHeld = ppO
                for spT in range(len(trSent[trMap])):
                    ppT = spT
                    trMapOut.write(str(parLine) + '_' + str(ppT) + ' ' + str(trMap) + '_' + str(spT) + '\n')
                ppTHeld = ppT
                #then do same for trSent[trMap] sentence
#                 if i == 1:
#                     print 'THREE'
#                     print trLine
#                     print orLine

            orMapPrev = orMap
            trMapPrev = trMap
            
#         if i == 1:
#             print trLine
#             print orLine
        orOut.write(orLine + '\n')
        trOut.write(trLine + '\n')
        
        orOut.close()
        trOut.close()
        orMapOut.close() 
        trMapOut.close()
                	            

def fixMapping(zhDir,enDir,zhSuf,enSuf):
    i = 0
    for f in os.listdir(zhDir):
        m = re.match('(.*)\.parallel',f)
        if not m: continue
        file = open(os.path.abspath(zhDir + '/' + f)) 
        this = file.readline()
        par = file.readline()
        map = re.match('.+\/([^\/]+)',par.split()[2])
        parfile = map.group(1)
        presfile = m.group(1)
        out = open(presfile+'_enmapfix.txt','w')
        out.write(presfile + '  par: ' + parfile + '\n\n')
        if this.startswith('translated'):
            transdoc = presfile
            origdoc = parfile
            transdir = zhDir
            origdir = enDir
            t = 1
        elif this.startswith('original'):
            origdoc = presfile
            transdoc = parfile
            transdir = enDir
            origdir = zhDir
            t = 0
        file.close()
        i += 1
#         print this
#         print presfile
#         print parfile
#         print origdoc
#         print transdoc
#         print '\n'
        zhSent = extractSentFromParse(os.path.abspath(zhDir + '/' + presfile + '.parse'))
        enSent = extractSentFromParse(os.path.abspath(enDir + '/' + parfile + '.parse')) 
        zhSentFix = extractSentFromParseFix(os.path.abspath(zhDir + '/' + presfile + '.parse'))
        enSentFix = extractSentFromParseFix(os.path.abspath(enDir + '/' + parfile + '.parse')) 
        
        for sp in range(len(enSent)):
            if len(enSent[sp]) != len(enSentFix[sp]):
                for wp in range(min([len(enSent[sp]),len(enSentFix[sp])])):
                    if enSent[sp][wp] != enSentFix[sp][wp]:
                        out.write(str(sp) + '_' + str(wp) + ' ' + enSentFix[sp][wp]+ '\n')
                        # for w in enSentFix[sp]: out.write(w + ' ')
#                         out.write('\n')
#                         for w in enSent[sp]: out.write(w + ' ')
#                         out.write('\n')
                        break
        out.close()           

        
def extractSentFromParse(file):
    sentences = []
    s = []
    for line in open(file):
        t = re.match('\(TOP.+',line)
        if t: 
            if len(s) > 0:
                sentences.append(s)
                s = []
#         m = re.match('\s*(\(\-?[A-Z0-9\$\,\.\;\?\:]+\-?[A-Z0-9\$]*\-?\-?[A-Z0-9\$]*\-?\s)+([^\)]+).+',line)
        m = re.match('\s*.*\s([^\(\)]+)\)+',line)
        n = re.match('.*\-NONE\- .*',line)
        if m and not n: 
            word = m.group(1).lower()
            s.append(word)
    sentences.append(s)

    return sentences
    
def extractSentFromParseFix(file):
    sentences = []
    s = []
    for line in open(file):
        t = re.match('\(TOP.+',line)
        if t: 
            if len(s) > 0:
                sentences.append(s)
                s = []
#         m = re.match('\s*(\(\-?[A-Z0-9\$\,\.\;\?\:]+\-?[A-Z0-9\$]*\-?\-?[A-Z0-9\$]*\-?\s)+([^\)]+).+',line)
        m = re.match('\s*.*\s([^\(\)]+)\)+',line)
        n = re.match('.*\-NONE\- .*',line)
        if m and not n: 
            word = m.group(1)
            s.append(word)
    sentences.append(s)

    return sentences
    
if __name__ == "__main__":
    writeParallelFiles(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])