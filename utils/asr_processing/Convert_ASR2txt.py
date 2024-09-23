import json
import os

# Path to your JSON file
json_file_path = r"F:\Ocr_for_AIC\Text_Video_Retrieval\utils\ASR\ASR_Output\ASR_Output\temp.json"

# Output txt file path
output_file_path = r'output.txt'

# Function to extract the required format
def process_image_paths(json_data):
    processed_lines = []
    for path in json_data:
        # Split the path to get the folder name and the image number
        parts = path.split('/')
        folder_name = parts[-2]  # L01_V001
        image_number = parts[-1].split('.')[0]  # 000482
        # Remove leading zeros from the image number
        image_number = image_number.lstrip('0')
        # Format: L01_V001, 482
        processed_line = f"{folder_name}, {image_number}"
        processed_lines.append(processed_line)
    return processed_lines

# Load the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Process the paths and write to the txt file
processed_data = process_image_paths(data)

# Write the output to a txt file
with open(output_file_path, 'w') as file:
    for line in processed_data:
        file.write(line + '\n')

print(f"Data has been saved to {output_file_path}")
