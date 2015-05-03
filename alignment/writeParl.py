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
        
        orOut = open(presfile + orSuf,'w')
        trOut = open(presfile + trSuf,'w')
            
        aFile = open(os.path.abspath(transdir + '/' + transdoc + '.parallel'))
        orLine = ''
        trLine = ''
        orMapPrev = -1
        trMapPrev = -1
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
#                 if i == 1: 
#                     print 'ONE'
#                     print trLine
#                     print orLine
            elif trMap == trMapPrev:
                orLine = orLine + ' ' + ' '.join(orSent[orMap])
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
                orLine = ' '.join(orSent[orMap])
                trLine = ' '.join(trSent[trMap])
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
                	            

            
        #***** now need to write sentences to files according to alignment, joining each sentence-level word list with spaces
        
def extractSentFromParse(file):
    sentences = []
    s = []
    for line in open(file):
        t = re.match('\(TOP.+',line)
        if t: 
            if len(s) > 0:
                sentences.append(s)
                s = []
        m = re.match('\s*(\(\-?[A-Z0-9]+\-?[A-Z0-9]*\-?\-?[A-Z0-9]*\-?\s)+([^\)]+).+',line)
        if m: 
            word = m.group(2)
            s.append(word)
    sentences.append(s)

    return sentences
    
if __name__ == "__main__":
    writeParallelFiles(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])