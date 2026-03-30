import os
import shutil

# 数据集路径
dataset_dir = '/data/tpy/projects/GS-CLIP-main/datasets/Anomaly-ShapeNet/'

fixed_count = 0
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.png'):
            file_path = os.path.join(root, file)
            # 检查文件大小是否为 0 字节
            if os.path.getsize(file_path) == 0:
                print(f"发现损坏文件: {file_path}")

                # 在同目录下找一个大于 0 字节的正常 png 文件来替换
                valid_replacement = None
                for other_file in files:
                    other_file_path = os.path.join(root, other_file)
                    if other_file.endswith('.png') and os.path.getsize(other_file_path) > 0:
                        valid_replacement = other_file_path
                        break

                if valid_replacement:
                    shutil.copy2(valid_replacement, file_path)
                    print(f"  --> 已使用 {os.path.basename(valid_replacement)} 覆盖")
                    fixed_count += 1
                else:
                    print(f"  --> 警告：该目录下找不到任何正常的替代图片！")

print(f"修复完成！共自动替换了 {fixed_count} 个损坏图片。")