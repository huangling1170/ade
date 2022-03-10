import os

root =r"E:\AIdata\yellow_man2\TRAIN"
img_name = os.listdir(root)
img_path = f"{root}/{img_name}"
print(img_path)
num,_= img_name.split(".")[0]
print(num)


