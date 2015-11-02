import re

questions = 'toefl.qst'
answers = 'toefl.ans'

prompt = {}
choices = {}
qfile = open(questions)
for line in qfile:
    if len(line.split()) == 0: continue
    m = re.match('([0-9]+)',line)
    l = re.match('([a-z]+)',line)
    if m: 
        num = m.group(1)
        prompt[num] = line.split()[1]
        choices[num] = {}
    elif l:
        let = l.group(1)
        choices[num][let] = line.split()[1]   
qfile.close()

ans = {}

afile = open(answers)
for line in afile:
    list = line.split()
    if len(list) == 0: continue
    ans[list[0]] = list[3]
afile.close()

with open('toefl.formatted','w') as out:
    for i in range(len(ans)):
        n = str(i+1)
        corr = choices[n][ans[n]]
        other = [item[1] for item in choices[n].items()if item[0] != ans[n]]
        strlist = [prompt[n],corr] + other
        out.write(' | '.join(strlist) + '\n')
