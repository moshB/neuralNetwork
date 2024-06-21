import cv2
import numpy as np
import matplotlib
import os
import random
from decimal import Decimal, getcontext

# Set the precision to 200 digits
getcontext().prec = 20

from matplotlib import pyplot as plt

# Specify Matplotlib backend
matplotlib.use('TkAgg')  # Change to another backend if needed


def generate_random_circle(image_size):
  """
  This function generates a random circle image of the specified size.

  Args:
      image_size (int): The size (width and height) of the desired image.

  Returns:
      numpy.ndarray: A 2D NumPy array representing the circle image, where 1 represents
                     filled pixels and 0 represents empty pixels.
  """

  # Create an empty image array with all zeros
  image = np.zeros((image_size, image_size), dtype=np.uint8)  # Adjust dtype if necessary

  # Ensure radius is within image bounds for proper circle placement
  max_radius = min(image_size // 2 - 1, 255)  # Consider buffer for edge pixels

  # Generate random center coordinates within the valid range
  center_x = random.randint(max_radius + 1, image_size - max_radius - 1)
  center_y = random.randint(max_radius + 1, image_size - max_radius - 1)

  # Generate random radius within the valid range
  radius = random.randint(1, max_radius)  # Ensure a minimum radius of 1

  # Iterate through each pixel and check if it falls within the circle
  for y in range(image_size):
    for x in range(image_size):
      distance_from_center = np.sqrt(((x - center_x) ** 2) + ((y - center_y) ** 2))
      if distance_from_center <= radius:
        image[y, x] = 1  # Mark pixel as filled if inside the circle

  return image
def generate_random_trapezoid(image_size):
    """
  Generates a random trapezoid within a square image of specified size.

  Args:
      image_size: The size of the square image (width and height).

  Returns:
      A 2D NumPy array representing the image with 0s and 1s indicating empty and filled pixels.
  """
    # Create an empty image
    img = np.zeros((image_size, image_size), dtype=np.uint8)

    # Generate random coordinates for bottom-left corner within image bounds
    min_x, max_x = 0, image_size - 1
    min_y, max_y = 0, image_size - 1
    bottom_left_x = random.randint(min_x, max_x)
    bottom_left_y = random.randint(min_y, max_y)

    # Generate random width and height ensuring they fit within the image
    max_width = max_x - bottom_left_x + 1
    max_height = max_y - bottom_left_y + 1
    width = random.randint(1, max_width)
    height = random.randint(1, max_height)

    # Generate random shift value for top base (0 for rectangle, positive for trapezoid)
    max_top_shift = width // 2  # Limit shift to half the width for a reasonable trapezoid
    top_shift = random.randint(1, max_top_shift)

    # Calculate coordinates for top-left and top-right corners
    top_left_x = bottom_left_x
    top_left_y = bottom_left_y + height
    top_right_x = bottom_left_x + width
    top_right_y = top_left_y

    # Fill the bottom rectangle part
    for y in range(bottom_left_y, bottom_left_y + height):
        for x in range(bottom_left_x, bottom_left_x + width):
            img[y, x] = 1

    # Fill the top trapezoid part (with potential shift)
    for y in range(top_left_y, top_left_y - top_shift, -1):
        # Adjust x coordinates for the trapezoid's top base
        left_x = max(bottom_left_x,
                     top_left_x - (width - top_shift) // 2 + (x - top_left_y) // (height - top_shift) * (width // 2))
        right_x = min(top_right_x + (width - top_shift) // 2 - (x - top_left_y) // (height - top_shift) * (width // 2),
                      max_x)
        for x in range(int(left_x), int(right_x) + 1):
            img[y-1, x-1] = 1#todo

    return img


def generate_random_rectangle(image_size):
    """
  Generates a random rectangle within a square image of specified size.

  Args:
      image_size: The size of the square image (width and height).

  Returns:
      A 2D NumPy array representing the image with 0s and 1s indicating empty and filled pixels.
  """
    # Create an empty image
    img = np.zeros((image_size, image_size), dtype=np.uint8)

    # Generate random coordinates for top-left corner within image bounds
    min_x, max_x = 0, image_size - 1
    min_y, max_y = 0, image_size - 1
    top_left_x = random.randint(min_x, max_x)
    top_left_y = random.randint(min_y, max_y)

    # Generate random width and height ensuring they fit within the image
    max_width = max_x - top_left_x + 1
    max_height = max_y - top_left_y + 1
    width = random.randint(1, max_width)
    height = random.randint(1, max_height)

    # Fill the rectangle in the image
    for y in range(top_left_y, top_left_y + height):
        for x in range(top_left_x, top_left_x + width):
            img[y, x] = 1

    return img


# Example usage with rectangle generation

def generate_random_triangle(image_size):
    """
  Generates a random triangle within a square image of specified size.

  Args:
      image_size: The size of the square image (width and height).

  Returns:
      A 2D list representing the image with 0s and 1s indicating empty and filled pixels.
  """
    # Create an empty image
    img = [[0 for _ in range(image_size)] for _ in range(image_size)]

    # Generate random vertex coordinates within the image bounds
    v1_x, v1_y = random.randint(0, image_size - 1), random.randint(0, image_size - 1)
    v2_x, v2_y = random.randint(0, image_size - 1), random.randint(0, image_size - 1)
    v3_x, v3_y = random.randint(0, image_size - 1), random.randint(0, image_size - 1)

    # Find minimum and maximum coordinates for bounding box
    min_x = min(v1_x, v2_x, v3_x)
    max_x = max(v1_x, v2_x, v3_x)
    min_y = min(v1_y, v2_y, v3_y)
    max_y = max(v1_y, v2_y, v3_y)

    # Iterate through pixels within the bounding box
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Check if pixel is inside the triangle using barycentric coordinates
            if is_inside_triangle(x, y, v1_x, v1_y, v2_x, v2_y, v3_x, v3_y):
                img[y][x] = 1

    return img


def is_inside_triangle(x, y, v1_x, v1_y, v2_x, v2_y, v3_x, v3_y):
    """
  Helper function to check if a point is inside a triangle using barycentric coordinates.

  Args:
      x, y: Coordinates of the point to check.
      v1_x, v1_y, v2_x, v2_y, v3_x, v3_y: Coordinates of the triangle vertices.

  Returns:
      True if the point is inside the triangle, False otherwise.
  """
    # Calculate area of the whole triangle
    total_area = abs((v1_x * (v2_y - v3_y) + v2_x * (v3_y - v1_y) + v3_x * (v1_y - v2_y)) / 2)

    # Calculate area of each sub-triangle formed by the point and each vertex
    area_a = abs((x * (v2_y - v3_y) + v2_x * (v3_y - y) + v3_x * (y - v2_y)) / 2)
    area_b = abs((v1_x * (y - v3_y) + x * (v3_y - v1_y) + v3_x * (v1_y - y)) / 2)
    area_c = abs((v1_x * (v2_y - y) + v2_x * (y - v1_y) + x * (v1_y - v2_y)) / 2)

    # Check if the sum of sub-triangle areas equals the whole triangle area (within a tolerance)
    return (abs(total_area - (area_a + area_b + area_c)) < 1e-6)






def generate_dataset(num_images, image_size):
    print('generate images')
    dataset = []
    labels = []
    # r=num_images//10

    for i in range(num_images):
        if i%10000==0:
            print('|',end='')
        defect_type = np.random.choice(['triangle', 'rectangle', 'trapez'])
        if defect_type == 'triangle':
            img = generate_random_triangle(image_size)
            label = [0.0]  # 0 represents bubble
        elif defect_type == 'rectangle':
            img = generate_random_rectangle(image_size)
            label = [0.5]  # 1 represents scratch
        else:
            img = generate_random_circle(image_size)#todo generate_random_trapezoid
            label = [1.0]  # 2 represents dirt stain

        dataset.append(img)
        # show_image(img,'name')
        labels.append(label)
    print()
    return np.array(dataset), np.array(labels)



def show_image(img,name):
    """
  Displays the generated random triangle image.
  """
    plt.imshow(img, cmap="binary")  # Use binary colormap for clear visualization
    plt.title("Random "+str(name))
    plt.axis("off")  # Hide axes for cleaner presentation
    plt.show()


# Example usage
# image_size = 128
# img = generate_random_triangle(image_size)
# show_image(img)
# image_size = 20
# img = generate_random_trapezoid(image_size)
# show_image(img)
# print(img)
# for it in img:
#     print(it)
# s, t = generate_dataset(1, 10)
# for it in s:
#     print(it)
# for i in range(len(s)):
#     show_image(s[i],t[i])
# img = generate_random_circle(1000)
# show_image(img,23)