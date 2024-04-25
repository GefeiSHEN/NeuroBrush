from PIL import Image, ImageDraw
import os
import random
import mixbox

# Get the absolute path of the image file
image_path = os.path.abspath("./brush/brush_large_horizontal.png")
# print(image_path)
# StrokeBlender\brush\brush_large_horizontal.png

stroke1 = Image.open(image_path).convert("RGBA")
stroke1.save("./img/stroke1.png")
# Get the width and height of the image
width, height = stroke1.size

# Create a mask with transparent horizontal stripes
mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)

stripe_count = 80  # Adjust as needed
stripe_thickness = 1  # Adjust as needed
min_stripe_distance = 3
max_stripe_distance = 15

# Generate random distances between each stripe within the specified range
stripes = sorted([random.randint(0, height) for _ in range(stripe_count)])
stripes.sort()

for i in range(len(stripes)):
    # Randomize the distance between the stripes
    if i < len(stripes) - 1:
        next_y = stripes[i + 1]
        distance = random.randint(min_stripe_distance, max_stripe_distance)
        if next_y - stripes[i] < distance:
            stripes[i + 1] += distance - (next_y - stripes[i])

    # Draw the stripe
    y = stripes[i]
    y_end = min(y + stripe_thickness, height)  # Ensure y1 is within image bounds
    
    # Ensure y0 is less than y1
    y0 = min(y, y_end)
    y1 = max(y, y_end)
    
    draw.rectangle([(0, y0), (width, y1)], fill=255)

# Apply the mask to the original image
erased_stroke1 = Image.new("RGBA", (width, height), (0, 0, 0, 0))
erased_stroke1.paste(stroke1, (0, 0), mask)
erased_stroke1.save("./img/erased1.png")
# Make each black pixel transparent
pixels = erased_stroke1.load()
for y in range(height):
    for x in range(width):
        if pixels[x, y] == (0, 0, 0, 255):  # If the pixel is black
            pixels[x, y] = (0, 0, 0, 0)     # Set its alpha channel to 0

# Save or display the modified image
# stroke1.show()
# erased_stroke1.show()
erased_stroke1.save("./img/output_image.png")
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
# Create a mask of the non-transparent part of the erased stroke on the canvas
canvas_mask = Image.new("L", (canvas_width, canvas_height), 0)
canvas_pixels = canvas.load()
erased_stroke_pixels = erased_stroke1.load()

for y in range(canvas_height):
    for x in range(canvas_width):
        # Calculate the corresponding coordinates on the erased stroke image
        stroke_x = x - stroke1_position[0]
        stroke_y = y - stroke1_position[1]
        
        # Check if the calculated coordinates are within the bounds of the erased stroke image
        if 0 <= stroke_x < width and 0 <= stroke_y < height:
            # Check if the corresponding pixel in the erased stroke is non-transparent
            if erased_stroke_pixels[stroke_x, stroke_y][3] > 0:
                # Set the corresponding pixel in the mask to white
                canvas_mask.putpixel((x, y), 255)

# Save or display the canvas mask image
# canvas_mask.show()
canvas_mask.save("./img/canvas_mask.png")

# Create a list to store the coordinates where the mask is non-zero
nonzero_coordinates = []

# Iterate over the canvas mask to find non-zero coordinates
for y in range(canvas_height):
    for x in range(canvas_width):
        # Check if the pixel value in the mask is non-zero
        if canvas_mask.getpixel((x, y)) > 0:
            # Append the coordinates to the list
            nonzero_coordinates.append((x, y))

# Print or return the list of non-zero coordinates
# print("Non-zero coordinates on the canvas:", nonzero_coordinates)
# Create a list to store the coordinates and corresponding RGB values
coordinates_with_rgbs = []

# Iterate over the non-zero coordinates
for coord in nonzero_coordinates:
    # Get the RGB values of the corresponding coordinate on the original canvas
    rgb = canvas_pixels[coord[0], coord[1]]
    # Append the tuple containing both the coordinates and RGB values
    coordinates_with_rgbs.append((coord, rgb))

# Print or return the list of tuples
# print("Coordinates with RGB values on the original canvas:", coordinates_with_rgbs)
