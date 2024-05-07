


import os 

# data_dir = "./data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
data_dir = "./data/raw_data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/"

all_cases = os.listdir(data_dir)

for case_name in all_cases:
    case_dir = os.path.join(data_dir, case_name)

    for data_name in os.listdir(case_dir):

        if "-" not in data_name:
            continue
        new_name = data_name.split("-")[-1]

        new_path = os.path.join(case_dir, new_name)

        old_path = os.path.join(case_dir, data_name)

        os.rename(old_path, new_path)

        print(f"{new_path} 命名成功")

