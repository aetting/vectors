#!/usr/bin/python
# -*- coding: utf8 -*-

import re, os, sys

with open(os.path.abspath(sys.argv[1])) as f:
    with open(os.path.abspath(sys.argv[2]),'w') as out:
        r = f.read()
        r = re.sub('</.*>','\xe3\x80\x82',r)
        r = re.sub('<.*>','',r)
        r = re.sub('ⅹ','',r)
        r = re.sub('[,;:\.]','',r)
        r = re.sub('\xe3\x80\x81','',r)
        r = re.sub('\xe2\x85\xb8','',r)
        r = re.sub('\xe3\x80\x8d','',r)
        r = re.sub('\xe3\x80\x8c','',r)
        r = re.sub('\xe3\x80\x8e','',r)
        r = re.sub('[\(\)]','',r)
        r = re.sub('\n','',r)
#         r = re.sub('(\xe3\x80\x82)\s+(\xe3\x80\x82)','$1$2',r) 
        r = re.sub('\?\s*(\xe3\x80\x82)*','\?\n',r) 
        r = re.sub('\!\s*(\xe3\x80\x82)*','\!\n',r) 
        r = re.sub('(\xe3\x80\x82)+','\n',r)
        r = re.sub('\n\s+','\n',r) 
        out.write(r)

    