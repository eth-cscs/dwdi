#!/bin/bash

export SLURM_TIME_FORMAT=%Y/%m/%d-%H:%M:%S

# for jj in `seq $1 $2` ;do

sacct --noheader --parsable --format \
JobID,Cluster,ConsumedEnergyRaw,ConsumedEnergy,AllocTRES,ElapsedRaw,\
Elapsed,Start,End,User,Account,Group,JobName,NCPUS,AllocNodes,NNodes,NodeList,\
ReqNodes,NTasks,Partition,State \
-j $1

# done
