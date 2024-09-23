from tqdm import tqdm  # Thêm thư viện tqdm để hiển thị tiến độ

# Đường dẫn đến các tệp merged và tệp gốc
merged_L1_path = 'merged_L1.txt'
merged_L2_path = 'merged_L2.txt'
input_file_path = 'input_fixed.txt'  # Tệp gốc đã có dạng L01_V001,482,...
output_file_path = 'output.txt'  # Tệp đầu ra sau khi bổ sung nội dung cột 3

# Hàm đọc dữ liệu từ tệp merged và lưu vào từ điển
def load_merged_data(merged_file):
    data_dict = {}
    with open(merged_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Tách dòng thành các phần dựa trên dấu phẩy
            parts = line.strip().split(',')
            if len(parts) >= 3:
                key = f"{parts[0].strip()},{parts[1].strip()}"
                content = parts[2].strip()
                data_dict[key] = content
    return data_dict

# Tải dữ liệu từ hai tệp merged
merged_data_L1 = load_merged_data(merged_L1_path)
merged_data_L2 = load_merged_data(merged_L2_path)

# Hợp nhất dữ liệu từ cả hai tệp merged
merged_data = {**merged_data_L1, **merged_data_L2}

# Hàm xử lý tệp đầu vào và bổ sung nội dung cột 3
def process_input_file(input_file, output_file, merged_data):
    # Đếm số dòng trong tệp đầu vào để ước tính tiến độ
    total_lines = sum(1 for line in open(input_file, 'r', encoding='utf-8'))
    
    # Dùng tqdm để hiển thị thanh tiến độ khi xử lý tệp
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Đang xử lý", unit=" dòng"):
            parts = line.strip().split(',')
            key = f"{parts[0].strip()},{parts[1].strip()}"
            # Kiểm tra xem có nội dung bổ sung trong merged data hay không
            if key in merged_data:
                # Thêm nội dung từ merged vào dòng
                new_line = f"{line.strip()}, {merged_data[key]}"
            else:
                # Nếu không có nội dung bổ sung, giữ nguyên dòng
                new_line = line.strip()
            outfile.write(new_line + '\n')

# Xử lý tệp input và ghi ra tệp output với nội dung cột 3 được bổ sung
process_input_file(input_file_path, output_file_path, merged_data)

print(f"Dữ liệu đã được bổ sung và lưu vào {output_file_path}")
