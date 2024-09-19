import json

# Đường dẫn đến file .json
file_path = r'C:\Users\hoang\Desktop\ASR\dataL1\L01_V001.json'

# Đọc file .json như một file văn bản
with open(file_path, 'r', encoding='utf-8') as file:
    # Đọc nội dung từng dòng của file
    for line in file:
        # Xóa các ký tự xuống dòng và phân tách các phần tử bằng dấu phẩy
        parts = line.strip().split(',')
        
        # Kiểm tra nếu đủ 4 phần tử
        if len(parts) == 4:
            file_name = parts[0]   # Ví dụ: 'L13_V001'
            start_time = parts[1]  # Ví dụ: '4.839'
            end_time = parts[2]    # Ví dụ: '3.561'
            text = parts[3]        # Ví dụ: 'Kính chào quý vị chúng tôi rất vui được'
            
            # Xử lý hoặc in ra màn hình
            print(f"Tên file: {file_name}, Bắt đầu: {start_time}, Kết thúc: {end_time}, Nội dung: {text}")
        else:
            print("Dòng dữ liệu không đúng định dạng:", line)
