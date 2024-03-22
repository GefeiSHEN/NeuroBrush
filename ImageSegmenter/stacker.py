import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def stack_images_reversed(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.png') and not f.endswith('.mask.jpg')]
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

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(stacked_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return stacked_image

stacked_image = stack_images_reversed('Data/PaintResult')
# cv2.imwrite('result.jpg', stacked_image)