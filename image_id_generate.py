"""
Ravina Lad
creates images element in annotated JSON. 
checks if particular id is present in image subset repository and then it add that entry in annotated json
-- purpose --> The original JSON file was extensive, containing non-sequential image IDs. My objective was to work with a subset of images, necessitating the extraction of only a few specific images from the dataset.
"""

import os
import json

image_dir = "C:/Users/ravin/Desktop/OBJ_DET/GenISP_subset_srgb/"

# Accessing filenames from the directory using os.listdir()
file_names = os.listdir(image_dir)

# Create JSON elements only for existing files
json_elements = []
for file_name in file_names:
    file_path = os.path.join(image_dir, file_name)  # Full path to check if the file exists
    if os.path.isfile(file_path):  # Checking if the file exists
        # Generate JSON element
        json_element = {
            "id": file_name.split('.')[0],  # Assuming filename format is consistent
            "width": 3936,
            "height": 2624,
            "file_name": file_name,
            "license": 1
        }
        json_elements.append(json_element)

# Write the filtered JSON elements to a file
with open('filtered_json_elements.json', 'w') as file:
    json.dump(json_elements, file, indent=4)


# import json

# # List to store JSON elements
# json_elements = []

# # Iterate through the range of IDs from DSC_0555 to DSC_1236
# for i in range(555, 1237):
#     # Generate the ID name
#     id_name = f"DSC_{i:04d}"
    
#     # Create a dictionary for each JSON element
#     json_element = {
#         "id": id_name,
#         "width": 3936,
#         "height": 2624,
#         "file_name": f"{id_name}.jpg",
#         "license": 1
#     }
    
#     # Append the JSON element to the list
#     json_elements.append(json_element)

# # Write the JSON elements to a file
# with open('generated_json_elements.json', 'w') as file:
#     json.dump(json_elements, file, indent=4)
