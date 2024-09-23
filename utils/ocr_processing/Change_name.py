import os

# Đường dẫn đến thư mục root
root_folder = r"F:\Ocr_for_AIC\Text_Video_Retrieval\data\txt_L2"

# Lặp qua từng thư mục con trong root
for folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder)
    
    # Kiểm tra nếu đó là một thư mục (ví dụ: Keyframes_L01, Keyframes_L02, ...)
    if os.path.isdir(folder_path):
        # Lặp qua từng tệp trong thư mục con
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Đảm bảo chỉ xử lý các tệp txt
            if file_name.endswith(".txt"):
                # Đọc nội dung tệp
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Thay thế chuỗi Keyframes_L01, Keyframes_L02,... bằng tên tệp (file_name không có đuôi .txt)
                new_content = content.replace(folder, file_name.replace(".txt", ""))
                
                # Ghi lại nội dung đã thay thế vào cùng tệp
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)

print("Đã thay đổi nội dung các tệp thành công!")
