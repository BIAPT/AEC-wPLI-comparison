classdef ccSBATCH
    % Compute Canada SBATCH submission arguments
    properties
        % Local script, remote working directory (home, by default)
        localScript = 'generate_aec';
        workingDirectory = '.';

        % You must specify a compute account (def, rrg, rpp or ctb type)
        account = 'def-sblain';

        % Nodes, cpus per task, GPUs per node
        nodes = 20;
        ntasksPerNode = 1;  % Must stay at 1
        cpusPerTask = 10;   % To allow speedup on entire nodes
        gpusPerNode = 0;

        % Specify the memory per CPU
        memPerCPU = '10000'

        % Requested walltime
        walltime = '00:30:00'

        % You may use otherOptions to append a string to the qsub command
        % e.g.
        otherOptions = '--mail-user=yacine.mahdid@mail.mcgill.ca --mail-type=ALL --job-name=make-aec'
        
    end

    methods(Static)
        function job = submitTo(cluster)
            opt = ccSBATCH();
            optPool = 'matlabpool';
            releaseStr = version('-release');
            releaseDbl = str2double(releaseStr(1:4));
            if (releaseDbl >= 2016); optPool = 'pool'; end

            job = batch(cluster,    opt.localScript,     ...
                optPool,            opt.getNbWorkers(),  ...
                'CurrentDirectory', opt.workingDirectory ...
                );
        end
    end

    methods
        function nbWorkers = getNbWorkers(obj)
            % Automatic: the (nodes * cpusPerTask - 1) rule
            nbWorkers = obj.nodes * obj.cpusPerTask - 1;
        end

        function submitArgs = getSubmitArgs(obj)
            % Account
            slurmAccount = '';
            if size(obj.account) > 0
                slurmAccount = sprintf('--account=%s', obj.account);
            end

            % Compute resources (cpus, gpus, memory, time)
            compRes = sprintf('--nodes=%d --ntasks-per-node=%d --cpus-per-task=%d', ...
                obj.nodes, obj.ntasksPerNode, obj.cpusPerTask);

            if obj.gpusPerNode > 0
                compRes = sprintf('%s --gres=gpu:%d', ...
                    compRes, obj.gpusPerNode);
            end

            compRes = sprintf('%s --mem-per-cpu=%s --time=%s', ...
                compRes, obj.memPerCPU, obj.walltime);

            % Returned sbatch arguments
            submitArgs = sprintf('%s %s %s', ...
                slurmAccount, compRes, obj.otherOptions);
        end
    end
end