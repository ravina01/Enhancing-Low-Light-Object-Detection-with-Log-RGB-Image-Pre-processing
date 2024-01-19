"""
Ravina Lad
This file resizes srgb images
"""

import os
import cv2

# Function to resize images in a folder and save them to a new folder
def resize_images(input_folder, output_folder, width=640, height=480):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of files in the input folder
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    for image_file in image_files:
        # Read the image using OpenCV
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)

        if img is not None:
            # Resize the image
            resized_img = cv2.resize(img, (width, height))

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, resized_img)
            print(f"Resized and saved: {output_path}")
        else:
            print(f"Error reading file: {img_path}")

# Example usage:
input_folder_path = 'C:/Users/ravin/Desktop/OBJ_DET/GenISP_subset_srgb'
output_folder_path = 'C:/Users/ravin/Desktop/OBJ_DET/GenISP_resized_srgb'

# Call the function to resize images from input_folder and save them to output_folder
resize_images(input_folder_path, output_folder_path)