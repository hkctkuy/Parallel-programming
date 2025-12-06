#!/bin/bash -eu

t=0:3

gpu () {

N=$1
px=$2
py=$3
pz=$4
pn=$(($px * $py * $pz))
gn=$pn
echo MPI Process number: $pn, GPU number: $gn, Decomposition params: $px, $py, $pz, Size: $N
bsub -W $t \
     -o GPU-N${N}-p${pn}-g${gn}.out \
     -e GPU-N${N}-p${pn}-g${gn}.err \
     -m "polus-c3-ib polus-c4-ib" \
     -R "span[hosts=1]" \
     -gpu "num=$gn:mode=shared:j_exclusive=yes" \
     -n $pn \
     mpiexec ./solver $N $px $py $pz gpu

}

cpu () {

N=$1
tn=$2
px=$3
py=$4
pz=$5
pn=$(($px * $py * $pz))
echo MPI Process number: $pn, Tread number: $tn, Decomposition params: $px, $py, $pz, Size: $N
OMP_PLACES=cores \
OMP_PROC_BIND=close \
mpisubmit.pl -W $t \
	     --stdout CPU-N${N}-p${pn}-t${tn}.out \
	     --stderr CPU-N${N}-p${pn}-t${tn}.err \
	     -p $pn -t $tn \
	     ./solver -- $N $px $py $pz
}

make

rm -rf *.out *.err core.*
bkill 0 || true

N=256

for i in $(seq 0 1); do
	# Serial
    cpu $N 1 1 1 1
    # 20 MPI
    cpu $N 1 5 2 2
    # 20x8 MPI/OpenMP
	cpu $N 8 5 2 2
	# 1 GPU MPI/CUDA
	gpu $N   1 1 1
	# 2 GPU MPI/CUDA
	gpu 256  2 1 1
    N=$(( 2 * $N ))
done
