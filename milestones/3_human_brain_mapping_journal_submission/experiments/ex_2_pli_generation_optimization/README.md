# PLI Generation Optimization
In this second experiment we are optimizing the code from the first experiment to find out what is the best way of running the analysis on the cluster without having to wait a day to run the code. The initial foray using MATLAB2020a Parallele Engine was wonky at best. We need to cut the analysis into smaller independant pieces and run them simultaneously using a slurm file. This is what we will be attempting to optimize for in this analysis.

The optimal structure would look like what we have with the `generate_jobs.bash` + the `job.sl` files for the Python portion of the analysis (i.e. `step_2 and step_3`). However, we need to figure out how to cut the file so that it accepts as parameter the p_id and the epoch so that it know what to load. The output will be customized based on the inputs so that there will be no collision between workers. We also needs a job.sl file that will serve as a template for sbatching.

### Code Setup
- Bash file to generate the multiple jobs
- Slurm file that contains the nodes configuration for each jobs
- Matlab file containing the parallele code that need to be ran for each jobs

## run_pli_generation.bash
We have one central bash file that needs to be run once logged in the cluster in the right directory as follow:
```bash
./run_pli_generation.bash example_pli_analysis.sl
```

This will iteratively `sbatch` the slurm file using different parameters to generate X*40 cores analysis.
```bash
# batching loop
for p_id in ${P_IDS[@]}; do
    for epoch in ${EPOCHS[@]}; do
        echo "${p_id} for state ${epoch}"
        sbatch --export=P_ID=$p_id,EPOCH=$epoch $1
    done
done
```
This basically iterates over each permutation of the parameters and sbatch the analysis. The `$1` representing the input at index 1 in the bash invocation which is example_pli_analysis.sl.


## example_pli_analysis.sl
This is the name of the slurm file we can schedule to run on as many cores as possible using the `run_pli_generation.bash` script above.
For now, since the goal of this analysis is to figure out the best structure for the optimal batching of jobs, we only have one slurm file. 

The main part of the analysis is this snippet:
```bash
matlab -nodisplay -r "example_pli_analysis('$P_ID', '$EPOCH')"
```
Once we've loaded matlab, we can call it on the command line an run the example_pli_analysis as a function with the parameter `P_ID` and `EPOCH` that were set in the previous bash file. The `example_pli_analysis.m` has the same name as the slurm file, but it doesn't needs to.

## example_pli_analysis.m
This is the main part of the analysis. It contains the Matlab code that was cut down to add more flexibility so that it can run directly for a given `P_ID` and a given `EPOCH`. Since we are working without the Parallele Engine from MATLAB2020a we need to do the weird gymnastic in which we set the `NUM_CPU=40` otherwise MATLAB will think that we want only 12 CPUS no matter what we set in the slurm file.

```matlab
    NUM_CPU = 40;
    % Disable this feature
    distcomp.feature( 'LocalUseMpiexec', false ) % This was because of some bug happening in the cluster
    % Create a "local" cluster object
    %local_cluster = parcluster('local')

    % Modify the JobStorageLocation to $SLURM_TMPDIR
    pc.JobStorageLocation = strcat('/scratch/yacine08/', getenv('SLURM_JOB_ID'))

    % Start the parallel pool
    parpool(NUM_CPU)
```

