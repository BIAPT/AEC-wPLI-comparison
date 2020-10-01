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
Once we've loaded matlab, we can call it on the command line an run the example_pli_analysis as a function with the parameter `P_ID` and `EPOCH` that were set in the previous bash file. The `example_pli_analysis.m` has the same name as the slurm file, but it doesn't need to. 

**clarification**: The `P_ID` and the `EPOCH` variable are string coming from two bash arrays of string that were set in the run_pli_generation.bash file which looks like this:
```bash
EPOCHS=("eyesclosed_1" "induction" "emergence_first" "emergence_last" "eyesclosed_8")
P_IDS=("MDFA03" "MDFA05" "MDFA06" "MDFA07" "MDFA10" "MDFA11" "MDFA12" "MDFA15" "MDFA17")
```
This means that if we want to run a subset of the whole analysis we can remove item from this list and we will not batch them for analysis. If we keep everything we will call `example_pli_analysis('$P_ID', '$EPOCH')` with 40 cores 5*9 times which will spawn 45 analysis.

## example_pli_analysis.m
This is the main part of the analysis. It contains the Matlab code that was cut down to add more flexibility so that it can run directly for a given `P_ID` and a given `EPOCH`. Since we are working without the Parallele Engine from MATLAB2020a we need to do the weird gymnastic in which we set the `NUM_CPU=40` otherwise MATLAB will think that we want only 12 CPUS no matter what we set in the slurm file.

```matlab
    % Disable this feature
    distcomp.feature( 'LocalUseMpiexec', false ) % This was because of some bug happening in the cluster
    % Create a "local" cluster object
    local_cluster = parcluster('local')

    % Modify the JobStorageLocation to $SLURM_TMPDIR
    pc.JobStorageLocation = strcat('/scratch/yacine08/', getenv('SLURM_JOB_ID'))

    % Start the parallel pool
    parpool(local_cluster, str2num(getenv('SLURM_CPUS_ON_NODE')))
```

The rest of the script corresponds to what was there before except trimmed down to handle 1 input participant with 1 input condition. The main idea is that it will load the right dataset, calculate PLI properly only on that dataset and then save it using the input to create a filename.

## Note
- I had a lots of trouble with MATLAB and sbatching the slurm file, I kept getting the same crash for the parcluster call. Turns out that some file were corrupted! Check out the section [Running multiple parallel MATLAB jobs simultaneously](https://docs.computecanada.ca/wiki/MATLAB). Bottom line is that MATLAB is very unreliable on clusters and made me lose a considerable amount of time. Will make the switch to Python for this section of the analysis asap.