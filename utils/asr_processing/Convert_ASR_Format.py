import pandas as pd
import os

# Hàm tính toán frame index
def calculate_frame_indices(fps, start_time, duration):
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    return range(start_frame, end_frame + 1)

# Đường dẫn thư mục root
root_dir = r"C:\Users\hoang\Desktop\ASR"

# Đường dẫn tới thư mục Excel và DataL2
excel_dir = os.path.join(root_dir, 'ExcelL1')
datal2_dir = os.path.join(root_dir, 'dataL1')

# Lấy tất cả tên tệp JSON trong thư mục DataL2
json_files = [f for f in os.listdir(datal2_dir) if f.endswith('.json')]
if not json_files:
    raise FileNotFoundError("Không có tệp JSON nào trong thư mục DataL2")

# Lặp qua từng tệp JSON
for json_file_name in json_files:
    json_file_path = os.path.join(datal2_dir, json_file_name)
    output_rows = []  # Khởi tạo danh sách chứa kết quả cho từng tệp JSON
    
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        line_number = 0
        for line in json_file:
            line_number += 1
            parts = line.strip().split(',')
            
            if line_number == 1:
                # Lấy tên file Excel từ nội dung đầu tiên
                excel_file_name = parts[0] + '.csv'
                csv_file_path = os.path.join(excel_dir, excel_file_name)

                # Đọc tệp CSV tương ứng
                csv_df = pd.read_csv(csv_file_path)
                fps = csv_df['fps'].iloc[0]  # Lấy giá trị FPS từ cột "FPS"

            if len(parts) == 4:
                file_name = parts[0]
                start_time = float(parts[1])
                duration = float(parts[2])
                content = parts[3]

                # Kiểm tra xem duration có phải là NaN không
                if pd.isna(duration):
                    print(f"Dòng {line_number} trong tệp {json_file_name} có duration là NaN, bỏ qua.")
                    continue  # Bỏ qua dòng này nếu duration là NaN
                
                # Tính toán các frame index
                frame_indices = calculate_frame_indices(fps, start_time, duration)
                
                # Tạo các hàng đầu ra
                for frame_idx in frame_indices:
                    output_rows.append([file_name, frame_idx, content])
            else:
                print(f"Dòng {line_number} trong tệp {json_file_name} không đúng định dạng: {line}")

    # Ghi kết quả ra tệp TXT với tên tương ứng
    if output_rows:
        output_file_name = parts[0] + '.txt'  # Tạo tên tệp dựa trên parts[0]
        output_file_path = os.path.join(root_dir, 'ASR_Output/L1', output_file_name)

        # Tạo thư mục ASR_Output nếu chưa tồn tại
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, 'w', encoding='utf-8') as txt_file:
            for row in output_rows:
                txt_file.write(','.join(map(str, row)) + '\n')

        print(f"Kết quả đã được lưu tại: {output_file_path}")
