
import glob 
import random 
import json 

def get_train_val_test_list_from_fulldata(data_dir, train_rate=0.7, val_rate=0.1, test_rate=0.2, seed=42):
    all_paths = glob.glob(f"{data_dir}/*.npz")

    ## eliminate the pre
    all_paths_save = []
    for p in all_paths:
        all_paths_save.append(p.split("/")[-1])
    all_paths = all_paths_save
    train_number = int(len(all_paths) * train_rate)
    val_number = int(len(all_paths) * val_rate)
    test_number = int(len(all_paths) * test_rate)
    random.seed(seed)
    random.shuffle(all_paths)
    train_datalist = all_paths[:train_number]
    val_datalist = all_paths[train_number: train_number + val_number]
    test_datalist = all_paths[-test_number:] 

    print(f"training data is {len(train_datalist)}")
    print(f"validation data is {len(val_datalist)}")
    print(f"test data is {len(test_datalist)}", sorted(test_datalist))

    datalist = {
        "train": train_datalist,
        "validation": val_datalist, 
        "test": test_datalist
    }

    datalist = json.dumps(datalist)

    with open("./data_split.json", "w") as f:
        f.write(datalist)
