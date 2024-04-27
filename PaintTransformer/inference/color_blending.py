from PIL import Image
from tkinter import filedialog
from random import randint, random
from random import sample
from math import floor
import math
import cv2
import mixbox
import numpy as np
from PIL import ImageFilter
from PerlinNoise import PerlinNoiseFactory
class ColorBlending:
    
    @staticmethod
    def dilate_mask(mask, kernel_size):
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
        # mask.show()
        # Convert binary mask to OpenCV format
        # mask_cv = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)

        # Define dilation kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Dilate the mask
        dilated_mask_cv = cv2.dilate(mask_array, kernel, iterations=9)

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
    @staticmethod
    def apply_mask_to_image(image, mask):
        """
        Apply a mask to an image using alpha composition.

        Args:
        image (PIL.Image): Image to apply the mask to.
        mask (PIL.Image): Mask image to be applied.

        Returns:
        PIL.Image: Image with the mask applied.
        """
        # Ensure the image has an alpha channel
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        if mask.mode != 'RGBA':  # 'L' mode for grayscale
            mask = mask.convert('RGBA')
        
        # Create a blank image with transparent background
        result = Image.new('RGBA', image.size, (0, 0, 0, 0))
        
        # Iterate over each pixel in the mask
        for x in range(mask.width):
            for y in range(mask.height):
                # Get the pixel value of the mask
                mask_pixel = mask.getpixel((x, y))
                # print(mask_pixel)
                # If the pixel in the mask is not dark
                if mask_pixel[2] > 0:
                    # Get the corresponding pixel from the original image
                    image_pixel = image.getpixel((x, y))
                    # Set the pixel in the result image to the color of the original image
                    result.putpixel((x, y), image_pixel)
        return result
    # def dilute_mask(mask,radius):
        
    #     return mask.filter(ImageFilter.GaussianBlur(radius))
    @staticmethod
    def blend_processing(img, patch):
        """
        Perform color blending between two images while considering transparency.

        Args:
        img (PIL.Image): Base image.
        patch (PIL.Image): Overlay image.

        Returns:
        PIL.Image: Blended image.
        """
        # Ensure both images have the same size
        assert img.size == patch.size, "Images must have the same dimensions"

        # Create a new blank image for the result
        result = Image.new("RGBA", img.size)

        # Iterate over each pixel of the images and blend them
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                # Get the pixel values of both images
                color1 = img.getpixel((x, y))
                color2 = patch.getpixel((x, y))

                # Check if the pixel in the overlay image is fully opaque
                if color2[3] == 255:
                    # Blend the colors only if the overlay pixel is fully opaque
                    blended_color = tuple(mixbox.lerp(color1[:3], color2[:3], 0.35)) + (255,)
                else:
                    # Use the color from the base image if the overlay pixel is transparent
                    blended_color = color1
                
                # Set the blended color to the result image
                result.putpixel((x, y), blended_color)

        return result
    def box_blur(img):
        """
        Apply a box blur with enhanced color blending to the image.

        Args:
        img (PIL.Image): Input image.

        Returns:
        PIL.Image: Blurred image.
        """
        # Convert PIL image to numpy array
        img = np.array(img)

        # Get image dimensions
        height, width, _ = img.shape

        # Iterate over each pixel in the image
        for y in range(height):
            for x in range(width):
                # Check if the pixel has alpha channel = 0
                if img[y, x, 3] == 0:
                    continue  # Skip calculation for transparent pixels

                # Initialize blended color
                blended_color = np.zeros(3, dtype=np.float64)

                # Accumulate neighboring pixel colors with blend effect
                for ny in range(y - 5, y + 5):
                    for nx in range(x - 5, x + 5):
                        # Check if the neighboring pixel is within bounds and not transparent
                        if 0 <= ny < height and 0 <= nx < width and img[ny, nx, 3] > 0:
                            # Apply blending with mixbox.lerp
                            blended_color = mixbox.lerp(blended_color, img[ny, nx, :3], 0.25)

                # Assign the blended color back to the pixel
                img[y, x, :3] = blended_color

        # Convert the numpy array back to a PIL image
        blurred_image = Image.fromarray(np.uint8(img))

        return blurred_image


        
    @staticmethod
    def color_blending(patches, result,current_canvas):
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
        # Convert current_canvas to RGBA mode if it's not already
        if current_canvas.mode != 'RGBA':
            current_canvas = current_canvas.convert('RGBA')

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
                # patches.show()
                
                
        mask=ColorBlending.creating_mask(patches)
        # mask.save("./img/mask_patch.png")
        mask = mask.convert("L")
        # mask.show()
        # mask after dilation
        dilated_mask=ColorBlending.dilate_mask(mask,3)
        # dilated_mask.save("./img/dilated_mask.png")
        masked_img=ColorBlending.apply_mask_to_image(img,dilated_mask)
        # masked_img.save("./img/masked_img.png")
        masked_patch=ColorBlending.apply_mask_to_image(patches,mask)
        # masked_patch.save("./img/masked_patch.png")
        # Apply sharpening filter
        # sharpen_kernel = np.array([[-1, -1, -1],
        #                         [-1, 9, -1],
        #                         [-1, -1, -1]])
        # masked_patch = masked_patch.filter(ImageFilter.SHARPEN)

        blended=ColorBlending.blend_processing(masked_img,masked_patch)
        # blended.save("./img/blended1.png")
        
        # Composite images using the diluted mask as an alpha channel
        # blended = Image.composite(patches, patches, mask)
        # blended.show()
        # patches.show()
        blended2=ColorBlending.box_blur(blended)
        result = Image.alpha_composite(current_canvas, blended2.convert("RGBA"))

        return result