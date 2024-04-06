from PIL import Image

import os
import mixbox

# # Get the absolute path of the image file
image_path = os.path.abspath("StrokeBlender/brush/brush_large_horizontal.png")

# Check if the image path contains "horizontal"
ishorizontal = "horizontal" in image_path.lower()

stroke1 = Image.open(image_path)
patch1 = stroke1.convert("RGBA")

# Get the dimensions of the patch
width, height = patch1.size

# Define the center of the patch
center_x = width // 2
center_y = height // 2

#define the color of patch 1
color1=R1,G1,B1=255,255,0
# Iterate over each pixel of the patch
for x in range(width):
    for y in range(height):
        # Get the pixel and its RGB values
        pixel = patch1.getpixel((x, y))
        red, green, blue, alpha = pixel
        
        # If the pixel is black (R, G, B values are all 0), make it transparent
        if (red, green, blue) == (0, 0, 0):
            patch1.putpixel((x, y), (0, 0, 0, 0))  # Set alpha channel to 0
        
        # Otherwise, set the pixel to yellow
        else:
            patch1.putpixel((x, y), color1+ (255,))  # Yellow color with full opacity


# patch1.show()

# Dimensions of the canvas
canvas_width = 800
canvas_height = 600

# Create a new blank canvas
canvas = Image.new("RGB", (canvas_width, canvas_height), color="black")

# Save the canvas in JPEG format
canvas.save("canvas.jpg")

# canvas.show()
canvas = Image.open("canvas.jpg")

stroke1_position=(100,100)

# canvas_color1=canvas.getpixel(stroke1_position)
# print(canvas_color1)

# t = 0.8              # mixing ratio

# rgb_mix = mixbox.lerp(color1,canvas_color1,t)
# print(rgb_mix)
# for x in range(width):
#     for y in range(height):
#         # Get the pixel and its RGB values
#         pixel = patch1.getpixel((x, y))
#         red, green, blue, alpha = pixel
        
#         # If the pixel is black (R, G, B values are all 0), make it transparent
#         if (red, green, blue) == (0, 0, 0):
#             patch1.putpixel((x, y), (0, 0, 0, 0))  # Set alpha channel to 0
        
#         # Otherwise, set the pixel to yellow
#         else:
#             patch1.putpixel((x, y), rgb_mix+(255,))  # Yellow color with full opacity

# patch1.show()

# Paste the stroke1 onto the canvas at the specified position
# canvas_color = canvas.getpixel(stroke1_position)
canvas.paste(patch1, stroke1_position, patch1)

canvas.show()

