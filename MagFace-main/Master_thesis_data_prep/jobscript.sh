#!/bin/sh
### General options
### ?- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J testjob_magface
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- set span if number of cores is more than 1
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 18:00
### request x GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s174139@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o hpc_out/%J/pipe_out.out
#BSUB -e hpc_out/%J/pipe_err.err
# -- end of LSF options --

### Load modules
#module load python3

mkdir -p hpc_out/$LSB_JOBID

#echo "Running the script..."
module load python3/3.10.13
#module load gcc/11.4.0-binutils-2.40
module load cuda/11.7


### Run setup
# sh setup.sh $run_dir || exit 1
source /work3/s174139/best_master_remote/bin/activate
echo $PWD

export TRANSFORMERS_CACHE=/work3/s174139
export HF_HOME=/work3/s174139
export TORCH_HOME=/work3/s174139


### Run python script

../run/run_fine_tuner.sh