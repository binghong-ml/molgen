#!/bin/bash
#SBATCH --job-name=frag
#SBATCH --partition=mbzuai
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/projects/mbzuai/peterahn/workspace/molgen/resource/log/job_%j.log
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=p1-r05-b.g42cloud.net

srun \
  --container-image=sungsahn0215/mol-hrl:latest \
  --no-container-mount-home \
  --container-mounts="/nfs/projects/mbzuai/peterahn/workspace/molgen:/molgen" \
  --container-workdir="/molgen/src" \
  bash ../script/${1}