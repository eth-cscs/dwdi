# Slurm

## sacct

`sacct.sh` can extract energy data from slurmDB together with other
informations of 1 or more job(s). For example:

```
./sacct.sh 38113
```

will report:

|JobID       |Cluster|ConsumedEnergyRaw|ConsumedEnergy|AllocTRES                                                |ElapsedRaw|Elapsed |Start              |End                |User    |Account|Group  |JobName     |NCPUS|AllocNodes|NNodes|NodeList                        |ReqNodes|NTasks|Partition|State    |
|------------|-------|-----------------|--------------|---------------------------------------------------------|----------|--------|-------------------|-------------------|--------|-------|-------|------------|-----|----------|------|--------------------------------|--------|------|---------|---------|
|38113       |santis |20543125         |20.54M        |billing=1152,cpu=1152,energy=20543125,mem=1840000M,node=4|2102      |00:35:02|2024-03-15-20:24:16|2024-03-15-20:59:18|piccinal|       |csstaff|eff.slm     |1152 |4         |4     |nid[001002,001004,001012,001014]|4       |      |normal   |COMPLETED|
|38113.batch |santis |5094129          |5.09M         |cpu=288,mem=460000M,node=1                               |2102      |00:35:02|2024-03-15-20:24:16|2024-03-15-20:59:18|        |       |       |batch       |288  |1         |1     |nid001002                       |1       |1     |         |COMPLETED|
|38113.extern|santis |20543125         |20.54M        |billing=1152,cpu=1152,mem=1840000M,node=4                |2102      |00:35:02|2024-03-15-20:24:16|2024-03-15-20:59:18|        |       |       |extern      |1152 |4         |4     |nid[001002,001004,001012,001014]|4       |4     |         |COMPLETED|
|38113       |santis |20536917         |20.54M        |cpu=1152,mem=1840000M,node=4                             |2101      |00:35:01|2024-03-15-20:24:16|2024-03-15-20:59:17|        |       |       |cuda-vars.sh|1152 |4         |4     |nid[001002,001004,001012,001014]|4       |16    |         |COMPLETED|

More details about sacct fields can be found in the `man sacct` manpage,
or in https://confluence.cscs.ch/display/KB/Alps / Accounting data / sacct

:warning: sometimes, sacct will quietly report wrong energy values.
