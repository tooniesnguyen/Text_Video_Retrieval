def search_in_file(file_path, search_text):
    # Mở file ở chế độ đọc
    with open(file_path, 'r', encoding='utf-8') as file:
        # Đọc toàn bộ nội dung của file
        lines = file.readlines()

    # Lọc và in các dòng chứa search_text
    for line in lines:
        if search_text in line:
            print(line.strip())

if __name__ == "__main__":
        # Đường dẫn đến file
    file_path = '/home/hoangtv/Desktop/Nhan_CDT/CERBERUS/research/Text_Video_Retrieval/data/temp/TXT_FILES/info_ocr_processed.txt'
    
    # Nhập chuỗi tìm kiếm từ người dùng
    import time
    start_time = time.time()
    search_text = "mimita tennies giáo văn thị thị thị marating marn satman"
    # Tìm kiếm và in kết quả
    search_in_file(file_path, search_text)
    print("Time execute: ", time.time()- start_time)
