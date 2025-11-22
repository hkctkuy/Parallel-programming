#!/bin/bash -eu

module load SpectrumMPI
module load OpenMPI
ulimit -s 10240 || true

make

rm -rf *.out *.err core.*
bkill 0 || true

N=128
px=2
py=2
pz=2
tn=1
t=0:3

for i in $(seq 0 3); do
	pn=$(($px * $py * $pz))
	echo Process number: $pn, Tread number: $tn, Decomposition params: $px, $py, $pz, Size: $N
	bsub -W $t \
	     -o N${N}-p${pn}-t${tn}.out \
	     -e N${N}-p${pn}-t${tn}.err \
	     -m "polus-c3-ib polus-c4-ib" \
	     -R "affinity[core($tn)]" \
             -R "span[hosts=1]" \
	     -n $pn \
	     OMP_NUM_THREADS=$tn \
	     mpiexec ./solver $N $px $py $pz
	# Test process increase
	# if [ $(( $i % 3 )) -eq 0 ]; then
	# 	px=$(($px * 2))
	# elif [ $(( $i % 3 )) -eq 1 ]; then
	# 	py=$(($py * 2))
	# else
	# 	pz=$(($pz * 2))
	# fi
	# Test thread increase
	tn=$(($tn * 2))
done
