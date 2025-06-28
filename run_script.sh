#!/bin/sh

set -x
set -e

exec=./build/TestHybridComp
EB0=1e-2
EB1=1e-3
EB2=1e-4

./build_script.sh

IN_DATA0=/home/leonli/SDRBENCH/single_precision/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32

$exec $IN_DATA0 float 3 512 512 512 $EB0

$exec $IN_DATA0 float 3 512 512 512 $EB1

$exec $IN_DATA0 float 3 512 512 512 $EB2



IN_DATA1=/home/leonli/SDRBENCH/single_precision/SDRBENCH-Hurricane-100x500x500/100x500x500/Pf48.bin.f32

$exec $IN_DATA1 float 3 500 500 100 $EB0

$exec $IN_DATA1 float 3 500 500 100 $EB1

$exec $IN_DATA1 float 3 500 500 100 $EB2



IN_DATA2=/home/leonli/SDRBENCH/double_precision/SDRBENCH-Miranda-256x384x384/density.d64

$exec $IN_DATA2 double 3 384 384 256 $EB0

$exec $IN_DATA2 double 3 384 384 256 $EB1

$exec $IN_DATA2 double 3 384 384 256 $EB2