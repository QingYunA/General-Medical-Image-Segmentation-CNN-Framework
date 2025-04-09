import os
from pathlib import Path

# 获取所有pred-*.nii.gz文件
files = Path('/disk/cyq/2025/diffusion_seg_compare/logs/densenet/predict-2025-04-08/14-34-50/pred_file').glob('pred-*.nii.gz')

for file in files:
    # 从文件名中提取数字
    print(file)
    number = file.name.split('-')[1].split('.')[0]
    print(number)
    # 转换为两位数格式
    new_number = str(int(number)).zfill(2)
    # 构建新文件名
    new_name = os.path.join(os.path.dirname(file), f'{new_number}.nii.gz')
    # 重命名文件
    os.rename(file, new_name)
    print(f'Renamed {file} to {new_name}')
