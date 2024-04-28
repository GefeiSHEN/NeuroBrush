import os
import shutil
import argparse

def process_folders(src_img_folder, src_mask_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    for subfolder in os.listdir(src_img_folder):
        subfolder_path = os.path.join(src_img_folder, subfolder)
        if os.path.isdir(subfolder_path):
            png_files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            if png_files:
                highest_png = max(png_files, key=lambda x: int(x.split('.')[0]))
                source_file = os.path.join(subfolder_path, highest_png)
                dest_file = os.path.join(dest_folder, f"{subfolder}.png")
                shutil.copy(source_file, dest_file)

    for mask_file in os.listdir(src_mask_folder):
        if mask_file.endswith('.tiff'):
            source_file = os.path.join(src_mask_folder, mask_file)
            dest_file = os.path.join(dest_folder, mask_file)
            shutil.copy(source_file, dest_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy images and masks to a destination folder.")
    parser.add_argument("src_img_folder", help="Path to the source image folder")
    parser.add_argument("src_mask_folder", help="Path to the source mask folder")
    parser.add_argument("dest_folder", help="Path to the destination folder")

    args = parser.parse_args()

    process_folders(args.src_img_folder, args.src_mask_folder, args.dest_folder)
