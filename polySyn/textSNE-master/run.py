#!/usr/bin/env python

import string, numpy, gzip
f = "/Users/allysonettinger/Desktop/SenseRetrofit-master/TEST.txt"
if f.endswith(".gz"):
    o = gzip.open(f, "rb")
else:
    o = open(f, "rb")
titles, x = [], []
o.readline()
for l in o:
    toks = string.split(l)
    titles.append(toks[0])
    x.append([float(f) for f in toks[1:]])
x = numpy.array(x)

#from tsne import tsne
from calc_tsne import tsne
#out = tsne(x, no_dims=2, perplexity=30, initial_dims=30, USE_PCA=False)
#out = tsne(x, no_dims=2, perplexity=30, initial_dims=30, use_pca=False)
out = tsne(x, no_dims=2, perplexity=30, initial_dims=80)

import render
render.render([(title, point[0], point[1]) for title, point in zip(titles, out)], "test-output2.rendered.png", width=3000, height=1800)
