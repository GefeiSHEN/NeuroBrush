import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def segment_image_by_depth(image_path, depth_map_path, n, expand=10):
    image = cv2.imread(image_path)
    sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
    image = cv2.filter2D(image, -1, sharpen_kernel)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    depth_map_normalized = depth_map / 255.0

    depth_ranges = np.linspace(0, 1, n + 1)
    segments = []
    masks = []
    for i in range(n):
        if i == n - 1:
            mask = (depth_map_normalized >= depth_ranges[i]).astype(np.uint8)
        else:
            mask = ((depth_map_normalized >= depth_ranges[i]) & (depth_map_normalized < depth_ranges[i+1])).astype(np.uint8)

        if expand > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (expand*2+1, expand*2+1))
            mask = cv2.dilate(mask, kernel)

        segment = cv2.bitwise_and(image, image, mask=mask)
        segments.append(segment)
        masks.append(mask)

    return segments, masks

# variance by number of pixels
def detail(image, mask):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    
    return lap.var() / np.count_nonzero(mask)

def saver(segments, masks, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for i, segment in enumerate(segments):
        segment_file_path = os.path.join(path, f'{i+1}.png')
        cv2.imwrite(segment_file_path, segment)
        mask_file_path = os.path.join(path, f'{i+1}_mask.tiff')
        cv2.imwrite(mask_file_path, masks[i]*255)
    
def visualizer(image_path, depth_map_path, segments):
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
        mask = np.any(segment != [0, 0, 0], axis=-1)
        stacked_segment[mask] = segment[mask]

    plt.subplot(2, n//2 + 3, n + 3)
    plt.imshow(cv2.cvtColor(stacked_segment, cv2.COLOR_BGR2RGB))
    plt.title('Stacked')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment an image by depth and save the segments.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('depth_map_path', type=str, help='Path to the depth map image.')
    parser.add_argument('n_segments', type=int, help='Number of segments.')
    parser.add_argument('expand', type=int, default=0, help='Pixels to expand the mask by.')
    parser.add_argument('output_path', type=str, help='Output path for saved segments.')


    args = parser.parse_args()

    # Segment the image by depth
    segments, masks = segment_image_by_depth(args.image_path, args.depth_map_path, args.n_segments, args.expand)

    score = [detail(segment, mask) for segment, mask in zip(segments, masks)]
    print(score)

    # Save the segments
    saver(segments, masks, args.output_path)

    # Optionally, visualize the segments
    visualizer(args.image_path, args.depth_map_path, segments)