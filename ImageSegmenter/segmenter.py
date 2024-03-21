import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_image_by_depth(image_path, depth_map_path, n):
    image = cv2.imread(image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    depth_map_normalized = depth_map / 255.0

    depth_ranges = np.linspace(0, 1, n + 1)
    segments = []
    for i in range(n):
        if i == n - 1:
            mask = (depth_map_normalized >= depth_ranges[i]).astype(np.uint8)
        else:
            mask = ((depth_map_normalized >= depth_ranges[i]) & (depth_map_normalized < depth_ranges[i+1])).astype(np.uint8)
        
        segment = cv2.bitwise_and(image, image, mask=mask)
        segments.append(segment)
    
    return segments

def saver(segments, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i, segment in enumerate(segments):
        segment_file_path = os.path.join(path, f'{i+1}.png')
        cv2.imwrite(segment_file_path, segment)
    
def visualizer(image, depth_map, segments):
    n = len(segments)
    image = cv2.imread(image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, n//2 + 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, n//2 + 3, 2)
    plt.imshow(depth_map, cmap='gray')
    plt.title('Depth Map')
    plt.axis('off')
    
    stacked_segment = np.zeros_like(image)

    for i, segment in enumerate(segments):
        plt.subplot(2, n//2 + 3, i + 3)
        plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        plt.title(f'Segment {i+1}')
        plt.axis('off')
        stacked_segment = cv2.addWeighted(stacked_segment, 1, segment, 1, 0)

    plt.subplot(2, n//2 + 3, n + 3)
    plt.imshow(cv2.cvtColor(stacked_segment, cv2.COLOR_BGR2RGB))
    plt.title('Stacked')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

n = 3
image_path = 'Data/Input/afgrl.jpg'
depth_map_path = 'Data/Input/afgrl_depth_16bit.png'
output_path = 'Data/Output/afgrl'
segments = segment_image_by_depth(image_path, depth_map_path, n)
saver(segments, output_path)
visualizer(image_path, depth_map_path, segments)
