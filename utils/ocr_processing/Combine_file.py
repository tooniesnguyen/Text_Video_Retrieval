import glob
import os
import argparse

def merge_txt_files(root_dir, output_file):
    # Lấy tất cả các tệp .txt trong tất cả các thư mục con
    input_files = sorted(glob.glob(os.path.join(root_dir, '**', '*.txt'), recursive=True))

    # Gộp nội dung tất cả các tệp .txt vào một tệp duy nhất
    with open(output_file, 'w') as output:
        for input_file in input_files:
            with open(input_file, 'r') as input:
                output.write(input.read())
            output.write('\n')
    print(f"Đã gộp xong các file vào {output_file}")

if __name__ == "__main__":
    # Tạo bộ phân tích đối số
    parser = argparse.ArgumentParser(description="Merge .txt files from subdirectories into a single file.")

    # Thêm đối số cho đường dẫn thư mục gốc và tệp đầu ra
    parser.add_argument('--root_dir', type=str, required=True, help="Path to the root directory containing .txt files.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output file where merged content will be saved.")

    # Phân tích đối số từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm merge với các đối số đã phân tích
    merge_txt_files(args.root_dir, args.output_file)
