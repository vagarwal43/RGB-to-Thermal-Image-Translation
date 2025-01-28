import os
import shutil

# Set the base directory where your 'train' folder is located.
source_dir = '/ocean/projects/cis220039p/ayanovic/datasets/train'  # Replace with the path to your 'train' folder
trainA_dir = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/trainA'  # Replace with your desired path for input_A
trainB_dir = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/trainB'  # Replace with your desired path for input_B

# Create target directories if they don't exist.
os.makedirs(trainA_dir, exist_ok=True)
os.makedirs(trainB_dir, exist_ok=True)

# Function to move files from source to target directory.
def move_files(src, tgt):
    files = os.listdir(src)
    for file in files:
        # shutil.move(os.path.join(src, file), tgt)
        shutil.copy(os.path.join(src, file), tgt)

# Loop through each sequence directory in 'train'.
for seq_dir in os.listdir(source_dir):
    print(f'Processing {seq_dir}...')
    seq_path = os.path.join(source_dir, seq_dir)  # seq_00_day .. seq_01_night .. seq_02_day .. seq_03_night

    # Check if the path is a directory.
    if os.path.isdir(seq_path):
        for subdir in os.listdir(seq_path):
            print(f'Processing {subdir}...')
            subdir_path = os.path.join(seq_path, subdir)  # 00 .. 01 .. 02 .. 03

            for fl in os.listdir(subdir_path):
                fl_path = os.path.join(subdir_path, fl)  # fl_rgb .. fl_ir_aligned

                # Move files from rgb to A.
                if fl == 'fl_rgb':
                    print(f'Processing {fl_path}...')
                    move_files(fl_path, trainA_dir)

                # Move files from ir_aligned to B.
                elif fl == 'fl_ir_aligned':
                    print(f'Processing {fl_path}...')
                    move_files(fl_path, trainB_dir)

print('Reorganization complete.')
