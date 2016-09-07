#!/bin/bash

#
# Script to run load test using the stress command
# collectl is used for capture monitoring data
#


DT=`date +%Y%m%d-%T`
LOG=load-$DT.log
SEC=20 # Duration for a 

SIM=false
QUICK=false

REPS=10  # Numbner of repitions

# Intit log file that tracks workloads
echo "date,time,m1,m2,m3,m4"  >> $LOG

# Don't execute stress commands
if [ "$1" == "sim" ]
then
   echo "Just show commands ..."
   SIM=true
else
   # Start collectl ....
   collectl -i 5 --iosize -oD -scdmnj  -P --sep ',' -f myfile.out &
   COLL_PID=$!
fi

# Here an options for a quick test
if [ "$1" == "quick" ]
then	
   echo "Very short test ..."
   SEC=10
   QUICK=true
   REPS=1
fi

# paramters for systems sizes
NP=`nproc`

# Base mem load: 0.5 phy mem
MEMTOT=`cat /proc/meminfo | grep MemTotal | awk '{print $2}'`
f=$(( ${MEMTOT}/2  ))

# Base load for hdd
BLHDD=$(( ${NP}*2 ))

# The tests

Tests[0]="--cpu"
Tests[1]="--vm-bytes ${f}K  --vm"
Tests[2]="--cpu 1  --hdd-bytes 10G  --hdd"

Flags[0]='1,0,0,0'
Flags[1]='0,1,0,0'
Flags[2]='0,0,1,0'

MAXCPU=$(( ${NP}*3  ))

# Run tests 
for t in {0..2}
do
  COUNTER=0
  while [  $COUNTER -lt $REPS ]
  do
    echo The counter is $COUNTER
    let COUNTER=COUNTER+1

    case $t in
    0)  f=`nproc` ;; # User nproc as factori
    1)  f=10 ;;
    2)  f=10 ;;
    *)  f=1 ;;
    esac
  
    # Three rounds of load
    for s in {1..3}
    do
      l=$(( ${s}*${f}  ))

      echo "Stage $s: ${Tests[${t}]} " $l

      # Log load status after 1st level 
      if [ ${s} -gt 1 ] 
      then
        echo "`date +%Y%m%d,%T`, ${Flags[${t}]}" >> $LOG
      else
        echo "`date +%Y%m%d,%T`, 0,0,0,0" >> $LOG
      fi

      if [ $SIM = true ]
      then
        echo stress  ${Tests[${t}]} $l  --timeout ${SEC}s
      else
         stress  ${Tests[${t}]} $l  --timeout ${SEC}s
      fi
    done

    # Now relax a bits ...
    for j in {1..2}
    do
      echo "Sleep $j times" 
      echo "`date +%Y%m%d,%T`, 0,0,0,0" >> $LOG
      if [ $SIM = true ]
      then
        echo sleep ${SEC}
      else
        sleep ${SEC}
      fi
    done

 done 
done

if [ $SIM == "false" ]
then
   echo "Try to stop collectl ..."
   kill $COLL_PID
fi 

