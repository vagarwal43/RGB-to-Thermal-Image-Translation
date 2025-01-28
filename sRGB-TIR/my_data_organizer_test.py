import os
import shutil

# Set the base directory where your 'train' folder is located.
source_dir = '/ocean/projects/cis220039p/ayanovic/datasets/test'  # Replace with the path to your 'train' folder
testA_dir = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/testA'  # Replace with your desired path for input_A
testB_dir = '/ocean/projects/cis220039p/ayanovic/vlr_project/sRGB-TIR/data/testB'  # Replace with your desired path for input_B

# Create target directories if they don't exist.
os.makedirs(testA_dir, exist_ok=True)
os.makedirs(testB_dir, exist_ok=True)

# Function to move files from source to target directory.
def move_files(src, tgt):
    files = os.listdir(src)
    for file in files:
        # shutil.move(os.path.join(src, file), tgt)
        shutil.copy(os.path.join(src, file), tgt)

# Loop through each sequence directory in 'train'.
for seq_dir in os.listdir(source_dir):
    print(f'Processing {seq_dir}...')
    seq_path = os.path.join(source_dir, seq_dir)  # day .. night

    # Check if the path is a directory.
    if os.path.isdir(seq_path):
        for subdir in os.listdir(seq_path):
            print(f'Processing {subdir}...')
            subdir_path = os.path.join(seq_path, subdir)  # ImagesIR .. ImagesRGB

            # Move files from ImagesRGB to A.
            if subdir == 'ImagesRGB':
                print(f'Processing {subdir_path}...')
                move_files(subdir_path, testA_dir)

            # Move files from ImagesIR to B.
            elif subdir == 'ImagesIR':
                print(f'Processing {subdir_path}...')
                move_files(subdir_path, testB_dir)

print('Reorganization complete.')
