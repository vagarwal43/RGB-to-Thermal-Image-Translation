#! /bin/bash

# parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
# parser.add_argument('--input_folder', type=str, help="input image folder")
# parser.add_argument('--output_folder', type=str, help="output image folder")
# parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# parser.add_argument('--seed', type=int, default=1, help="random seed")
# parser.add_argument('--num_style',type=int, default=30, help="number of styles to sample")
# parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
# parser.add_argument('--output_only', action='store_true', help="whether only save the output images or also save the input images")
# parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
# parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
# parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
# parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
# parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
# parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")

source $PROJECT/miniconda3/etc/profile.d/conda.sh
conda activate vlr

python '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/inference_batch.py' \
--config /ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/configs/tir2rgb_folder.yaml \
--input_folder /ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/testA \
--output_folder /ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/outputs_segmentation_rgb_only/tir2rgb_folder/inference_images_Freiberg \
--checkpoint /ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/outputs_segmentation_rgb_only/tir2rgb_folder/checkpoints/gen_00025000.pt \
--output_path /ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/outputs_segmentation_rgb_only/tir2rgb_folder \
--a2b 1 --seed 1 --num_style 1 \
--input_folder_ir /ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/testB \
--segmentation