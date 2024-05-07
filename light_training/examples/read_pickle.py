import pickle
 
f = "/home/xingzhaohu/jiuding_code/SegRap2023/data/fullres/train/segrap_0000.pkl"

with open(f, "rb") as ff:
    s = pickle.load(ff)

    print(s)