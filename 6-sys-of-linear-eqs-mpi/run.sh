#!/bin/bash -eu

make

rm -rf out*.out err*.err core.*
bkill 0 || true

nx=8000
ny=8000
k1=101
k2=57
px=1
py=1
maxit=1000
eps=0.0001
tn=1
ll=1
t=0:3

for i in $(seq 5); do
	pn=$(($px * $py))
	echo Process number: $pn, Tread number: $tn, Decomposition params: $px, $py, Data size: $(($nx * $ny))
	bsub -W $t \
		-o out-p$pn-t$tn.out \
		-e err-p$pn-t$tn.err \
		-m "polus-c3-ib polus-c4-ib" \
		-R "affinity[core($pn)]" \
		OMP_NUM_THREADS=$tn \
		mpiexec -n $pn \
		./solver $nx $ny $k1 $k2 $px $py $maxit $eps $tn $ll
	# Uncomment to test paralell acceleration for thread increase
	# tn=$(($tn * 2))
	# Uncomment to test paralell acceleration for process increase
	if [ $(( $i % 2 )) -eq 0 ]; then
		px=$(($px * 2))
	else
		py=$(($py * 2))
	fi
	# Uncomment to test memory and time dependence on data size
	# if [ $(( $i % 2 )) -eq 0 ]; then
	# 	nx=$(($nx * 10))
	# else
	# 	ny=$(($ny * 10))
	# fi
done
