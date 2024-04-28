import cv2
import os
import numpy as np
import argparse
from scipy.ndimage import distance_transform_edt

# Thank you hw4
def blendImagePair(img1, mask1, img2, mask2, mode='blend'):
    height, width, _ = img2.shape
    ret = np.zeros_like(img2).astype(float)
    
    if mode == 'blend':
        mask1 = distance_transform_edt(mask1)
        mask1 = mask1 / np.max(mask1)
        mask2 = distance_transform_edt(mask2)
        mask2 = mask2 / np.max(mask2)

    for i in range(height):
        for j in range(width):
            if mode == 'overlay':
                if mask1[i][j] > 0:
                    ret[i, j] = img1[i, j]
                elif mask2[i][j] > 0:
                    ret[i, j] = img2[i, j]
            else:
                if mask1[i, j] + mask2[i, j] > 0:
                    ret[i, j] = (img1[i, j] * mask1[i, j] + img2[i, j] * mask2[i, j]) / (mask1[i, j] + mask2[i, j])

    ret = ret.astype(np.uint8)
    return ret

def convert_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(float)

    mask[np.isnan(mask)] = 0
    mask[mask == 255] = 1
    mask[mask != 1] = 0
    mask = mask.astype(np.uint8)
    
    return mask


def stack_images_reversed(folder_path, output_path):
    img_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    img_files.sort(reverse=True)
    
    stacked_image = None
    last_image = None
    last_mask = None
    
    for filename in img_files:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        mask_path = os.path.join(folder_path, filename.replace('.png', '_mask.tiff'))
        mask = convert_mask(mask_path)

        image = cv2.bitwise_and(image, image, mask=mask)

        if last_image is not None and last_mask is not None:
            image = blendImagePair(last_image, last_mask, image, mask)
        
        last_image = image
        if last_mask is not None:
            last_mask = np.logical_or(mask, last_mask)
        else:
            last_mask = mask
        stacked_image = image

    cv2.imwrite(output_path, stacked_image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Stacker with mask')
    parser.add_argument('folder_path', type=str, help='image folder')
    parser.add_argument('output_path', type=str, help='output path')

    args = parser.parse_args()
    stack_images_reversed(args.folder_path, args.output_path)
