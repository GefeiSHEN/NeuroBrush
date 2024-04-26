from PIL import Image
from tkinter import filedialog
from random import randint, random
from random import sample
from math import floor
import math
import cv2
import numpy as np
from PIL import ImageFilter
from PerlinNoise import PerlinNoiseFactory
class ColorBlending:
    @staticmethod
    def dilate_mask(mask, kernel_size=3):
        """
        Dilate a binary mask.

        Args:
        mask (PIL.Image): Binary mask image.
        kernel_size (int): Size of the dilation kernel.

        Returns:
        PIL.Image: Dilated mask image.
        """
        # Convert PIL image to numpy array
        mask_array = np.array(mask)

        # Convert binary mask to OpenCV format
        mask_cv = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)

        # Define dilation kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Dilate the mask
        dilated_mask_cv = cv2.dilate(mask_cv, kernel, iterations=1)

        # Convert back to PIL image
        dilated_mask_pil = Image.fromarray(dilated_mask_cv)

        return dilated_mask_pil
    def creating_mask(patches):
        width, height = patches.size
        mask = Image.new("L", (width, height), 0)  # L mode for grayscale mask
        patches_data = patches.load()
        mask_data = mask.load()

        for x in range(width):
            for y in range(height):
                pixel = patches_data[x, y]
                # Handle images with and without an alpha channel
                if len(pixel) == 4:
                    r, g, b, a = pixel
                else:
                    r, g, b = pixel
                    a = 255  # Assume full opacity for images without alpha channel
                
                # Check if the pixel is black and fully opaque
                if r == 0 and g == 0 and b == 0 and a == 255:
                    mask_data[x, y] = 0
                else:
                    mask_data[x, y] = 255
        # mask.show()
        return mask
    
    def dilute_mask(mask,radius):
        
        return mask.filter(ImageFilter.GaussianBlur(radius))
    @staticmethod
    def color_blending(patches, result):
        """
        Perform color blending operation.

        Args:
        patches (PIL.Image): Patches to be blended.
        result (PIL.Image): Base image.

        Returns:
        PIL.Image: Blended image.
        """
        # Implement the color blending operation here
        img = result.convert("RGBA")
        flag=0
        width,height=img.size
        perlin = PerlinNoiseFactory(2, octaves=4)
        for y in range(height):
            for x in range(width):
                # Get Perlin noise value for the current pixel position
                noise_value = perlin(x / width, y / height)
                
                
                # Get the original pixel value
                pixel = patches.getpixel((x, y))
                # Calculate the new brightness based on the noise value
                new_brightness = int(pixel[0] * (1.1 + noise_value))
                
                # Create a new pixel with the same color channels but modified brightness
                new_pixel = (new_brightness, pixel[1], pixel[2])
                # If the mask pixel is black (i.e., has no content), set the corresponding pixel in the patches image as transparent
                if pixel == (0, 0, 0, 255):
                    patches.putpixel((x, y), (0, 0, 0, 0))
                # Update the pixel in the image
                patches.putpixel((x, y), new_pixel)
        # mask=ColorBlending.creating_mask(patches)
        # mask = mask.convert("L")
        # mask.show()
        # mask=ColorBlending.dilate_mask(mask,2)
        # mask.show()
        # Composite images using the diluted mask as an alpha channel
        # blended = Image.composite(patches, patches, mask)
        # blended.show()
        # patches.show()
        
        return patches
    
    
    


img=Image.open("./img/002.png")

patches=Image.open("./img/003_1.png")
# img.show()
result=ColorBlending.color_blending(patches,img)
result.show()