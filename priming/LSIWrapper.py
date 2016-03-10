import gensim, os, sys, gzip, re


class MyDocs(object):
    def __init__(self,dirlist):
        self.dirlist = dirlist

    def __iter__(self):
        for d in self.dirlist:
            for f in os.listdir(d):
                doc = []
                if f.endswith('.gz'): fileObject = gzip.open(os.path.join(d,f))
                else: fileObject = open(os.path.join(d,f))
                for line in fileObject:
                    if re.match('<\/DOC',line): 
                        yield doc
                        doc = []
                        continue
                    doc.append(line)

dirlist = [sys.argv[i] for i in range(2,len(sys.argv))]
saveto = sys.argv[1]
#for x in MySent(dirlist): z = 1
corpus = MyDocs(dirlist)
for x in corpus:
    print x
    print 'X\nX\nX\nX\nX\n'
#model = gensim.models.Word2Vec(sentences,min_count=10)
#model.save(saveto)
