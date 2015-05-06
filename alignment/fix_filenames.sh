for filename in *src.*; do echo mv $filename ${filename//src./}; done > sr.sh
for filename in *tgt.*; do echo mv $filename ${filename//tgt./}; done > tg.sh
for filename in *unseg.*; do echo mv $filename ${filename//unseg./}; done > un.sh

chmod u+x sr.sh
chmod u+x tg.sh
chmod u+x un.sh

./sr.sh
./tg.sh
./un.sh

rm sr.sh
rm tg.sh
rm un.sh