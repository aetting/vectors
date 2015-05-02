import re, os, sys

dir = sys.argv[1]
i = 0
for f in os.listdir(dir):
    m = re.match('(.*)\.parallel',f)
    if not m: continue
    file = open(os.path.abspath(dir + '/' + f)) 
    this = file.readline()
    par = file.readline()
    map = re.match('.+\/([^\/]+)',par.split()[2])
    parfile = map.group(1)
    presfile = m.group(1)
    if this.startswith('translated'):
        transdoc = presfile
        origdoc = parfile
        transsuf = '.zh'
        origsuf = '.en'
    elif this.startswith('original'):
        origdoc = presfile
        transdoc = parfile
        transsuf = '.en'
        origsuf = '.zh'
    file.close()
    print this
    print presfile
    print parfile
    print origdoc
    print transdoc
