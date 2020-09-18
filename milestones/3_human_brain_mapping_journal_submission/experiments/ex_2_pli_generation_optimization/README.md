# PLI Generation Optimization
In this second experiment we are optimizing the code from the first experiment to find out what is the best way of running the analysis on the cluster without having to wait a day to run the code. The initial foray using MATLAB2020a Parallele Engine was wonky at best. We need to cut the analysis into smaller independant pieces and run them simultaneously using a slurm file. This is what we will  be attempting.

The optimal structure would look like what we have with the `generate_jobs.bash` + the `job.sl` files for the Python portion of the analysis (i.e. `step_2 and step_3`). However, we need to figure out how to cut the file so that it accepts as parameter the p_id and the epoch so that it know what to load. The output will be customized based on the inputs so that would be fine there will be no collision between workers. We also needs a job.sl file that will serve as a template for sbatching.

## run_pli_generation.bash
We have one central bash file that needs to be run once logged in the cluster in the right directory as follow:
```bash
./run_pli_generation.bash [analysis_script.sl]
```

This will iteratively `sbatch` the slurm file using different parameters to generate X*40 cores analysis.

## example_pli.sl
This is the name of the slurm file we can schedule to run on as many cores as possible using the `run_pli_generation.bash` script above.
For now, since the goal of this analysis is to figure out the best structure for the optimal batching of jobs, we only have one slurm file. 

