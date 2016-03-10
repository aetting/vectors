import gensim, os, sys, gzip


class MySent(object):
    def __init__(self,dirlist):
        self.dirlist = dirlist

    def __iter__(self):
        for d in self.dirlist:
            for f in os.listdir(d):
                for line in gzip.open(os.path.join(d,f)):
                    if len(line.strip()) > 0 and not re.match('<\/doc>',line):
                        yield line.split()

dirlist = [sys.argv[i] for i in range(2,len(sys.argv))]
saveto = sys.argv[1]
sentences = MySent(dirlist)
for x in sentences: print x
#model = gensim.models.Word2Vec(sentences,min_count=10)
#model.save(saveto)
