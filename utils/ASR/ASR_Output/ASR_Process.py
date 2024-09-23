import json
import re

# Hàm loại bỏ các số 0 dư thừa trong frame idx
def remove_leading_zeros(frame_idx):
    return str(int(frame_idx))

# Hàm đọc file json
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Hàm cập nhật file merged và chỉ giữ lại các dòng có trong file json
def filter_merged_file(input_file, output_file, valid_entries):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Chỉ giữ lại các dòng có tên video và frame trong danh sách valid_entries
    new_lines = []
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 2:
            video_name = parts[0].strip()
            frame_idx = parts[1].strip()

            # Kiểm tra xem video_name và frame_idx có nằm trong danh sách hợp lệ không
            if (video_name, frame_idx) in valid_entries:
                new_lines.append(line)

    # Lưu các dòng hợp lệ vào file mới
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

# Đường dẫn tới file json
json_file = 'temp.json'
data = read_json(json_file)

# Danh sách để chứa các video_name và frame_idx hợp lệ từ file JSON
valid_entries = []

# Duyệt qua từng phần tử trong file JSON và trích xuất video_name và frame_idx
for path in data:
    match = re.search(r'images/(Keyframes_L\d+)/(\w+)/(\d+)\.jpg', path)
    if match:
        video_name = match.group(2)  # L01_V001
        frame_idx = remove_leading_zeros(match.group(3))  # 000482 -> 482
        valid_entries.append((video_name, frame_idx))

# Xác định file merged cần truy cập và lọc nội dung
for video_name, frame_idx in valid_entries:
    if 'L01' <= video_name.split('_')[0] <= 'L12':
        merged_file = 'merged_L1.txt'
        output_file = 'output_L1.txt'  # File kết quả cho merged_L1.txt
    else:
        merged_file = 'merged_L2.txt'
        output_file = 'output_L2.txt'  # File kết quả cho merged_L2.txt

    # Lọc nội dung trong file merged và chỉ giữ lại các dòng có trong valid_entries
    filter_merged_file(merged_file, output_file, valid_entries)
