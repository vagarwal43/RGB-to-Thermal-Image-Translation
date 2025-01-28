#! /bin/bash
source $PROJECT/miniconda3/etc/profile.d/conda.sh
conda activate vlr
python '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/train.py' \
--output_path ./outputs_segmentation_rgb_only \
--segmentation --resume