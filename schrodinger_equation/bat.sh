#!/bin/bash
make
make clean
folder="result"
if [ ! -d ${folder} ]; then
	mkdir ${folder}
fi
for (( i=-30;i<=10;i=i+1 ))
do
	echo "scale=1;${i}/10.0" | bc | python input.py
	./dvr >> output 2>>log
	python plot_psi.py &
	python plot_phase.py &
	wait
	for f in *.txt *.png *.gif input
	do
		mv -- "${f}" "${folder}/${i}.${f}"
	done
	echo "Finished 10.0 * lnE = ${i}.0"
	echo $(date +"%Y-%m-%d %H:%M:%S.%N")
done
mv output log ${folder}
