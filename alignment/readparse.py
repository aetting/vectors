import re

sentences = []
s = []
for line in open('/Users/allysonettinger/Desktop/vectors/alignment/en/cctv_0000.parse'):
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

print sentences 