#!/bin/bash

#SBATCH -p batch
#SBATCH -A coreai_dlalgo_nemorl
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --job-name=nemo-25.07-import  # 작업 이름도 변경하는 것을 권장합니다.
#SBATCH --output=nemo-25.07-import%j.log # 로그 파일 이름도 변경하는 것을 권장합니다.


# Run command
enroot import -o nemo_25.11.rc6.sqsh docker://nvcr.io/nvidian/nemo:25.11.rc6


