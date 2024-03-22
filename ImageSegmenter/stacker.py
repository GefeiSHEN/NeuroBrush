import cv2
import os
import numpy as np
import argparse

def stack_images_reversed(folder_path, output_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.png') and not f.endswith('.mask.tif')]
    files.sort()
    files.reverse()
    
    stacked_image = None
    
    for filename in files:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        if stacked_image is None:
            stacked_image = np.zeros_like(image)

        mask_name = f"{filename}.mask.tif"
        mask_path = os.path.join(folder_path, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        height, width, _ = stacked_image.shape

        for i in range(height):
            for j in range(width):
                if mask[i][j] != 0:
                    stacked_image[i, j] = image[i, j]

    cv2.imwrite(output_path, stacked_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Stacker with mask')
    parser.add_argument('folder_path', type=str, help='image folder')
    parser.add_argument('output_path', type=str, help='output path')

    args = parser.parse_args()
    stack_images_reversed(args.folder_path, args.output_path)