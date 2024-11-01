#!/bin/bash -eu
opt=$3
: '
clang++ solver.cc -o solver \
    -std=c++17 -O$opt \
    -fopenmp=libiomp5 \
    -I/usr/lib/gcc/x86_64-linux-gnu/11/include
'
g++ solver.cc -o solver \
    -std=c++17 -O$opt \
    -fopenmp \
    -I/usr/lib/gcc/x86_64-linux-gnu/11/include

nx=$1
ny=$2
k1=101
k2=57
maxit=1000
eps=0.0001
tn=1
ll=1
echo $1 $2
for i in $(seq 4); do
    echo Tread number: $tn
    ./solver $nx $ny $k1 $k2 $maxit $eps $tn $ll
    tn=$(($tn * 2))
done
