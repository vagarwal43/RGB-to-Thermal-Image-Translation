import os
import cv2
import shutil

def combine_images(IR_path, VI_path, output_path):
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get list of filenames in IR and VI paths
    filename_IR = sorted(os.listdir(IR_path))
    filename_VI = sorted(os.listdir(VI_path))

    # Pair images based on filenames
    paired_filenames = []
    for ir_filename in filename_IR:
        for vi_filename in filename_VI:
            if ir_filename == vi_filename:
                paired_filenames.append((vi_filename, ir_filename))
                break

    # Combine paired images and store in the output directory
    for ir_filename, vi_filename in paired_filenames:
        ir_filepath = os.path.join(IR_path, ir_filename)
        vi_filepath = os.path.join(VI_path, vi_filename)

        ir_image = cv2.imread(ir_filepath)
        vi_image = cv2.imread(vi_filepath)

        # Resize images to have the same height
        min_height = min(ir_image.shape[0], vi_image.shape[0])
        ir_image = cv2.resize(ir_image, (int(ir_image.shape[1] * min_height / ir_image.shape[0]), min_height))
        vi_image = cv2.resize(vi_image, (int(vi_image.shape[1] * min_height / vi_image.shape[0]), min_height))

        # Combine images side by side
        combined_image = cv2.hconcat([vi_image, ir_image])

        # Save the combined image
        combined_filename = ir_filename.split('.')[0] + '_combined.jpg'
        combined_filepath = os.path.join(output_path, combined_filename)
        cv2.imwrite(combined_filepath, combined_image)

    print("Combining complete.")

if __name__ == "__main__":
    # Provide paths to the IR and VI directories
    IR_path = "/home/dell/pytorch-CycleGAN-and-pix2pix/reduced_dataset/trainB"
    VI_path = "/home/dell/pytorch-CycleGAN-and-pix2pix/reduced_dataset/trainA"

    # Provide path to the output directory
    output_path = "/home/dell/pytorch-CycleGAN-and-pix2pix/reduced_dataset/train"

    # Combine images and store in the output directory
    combine_images(IR_path, VI_path, output_path)
