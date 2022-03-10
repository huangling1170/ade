import os

import PIL
from PIL import Image
import numpy as np

bg_path = "bg_pic"
x = 0
for filname in os.listdir(bg_path):
    x+=1
    try:
        background = Image.open("{0}/{1}".format(bg_path, filname))
        background_path = f"{bg_path}/{filname}"  # bg_pic/1.png

        shape = np.shape(background)
        if len(shape) == 3 and shape[0] > 100 and shape[1] > 100  :  # 通道不一定为3，shape[2] == 3 会报错
            background = background
        else:
            continue  # 跳过
        background_resize = background.resize((300, 300))
        background_resize = background_resize.convert("RGB")
        name = np.random.randint(1, 21)
        img_font = Image.open("yellow/{0}.png".format(name))
        ran_w = np.random.randint(50, 180)
        img_new = img_font.resize((ran_w, ran_w))

        ran_x1 = np.random.randint(0, 300 - ran_w)
        ran_y1 = np.random.randint(0, 300 - ran_w)

        r, g, b, a = img_new.split()
        font_path, num = background_path.split("/")
        num = num.split(".")[0]




        if 0 < int(num) <= 40000:
            background_resize.paste(img_new, (ran_x1, ran_y1), mask=a)

            ran_x2 = ran_x1 + ran_w
            ran_y2 = ran_y1 + ran_w

            background_resize.save(r"E:/AIdata/yellow_man2/TRAIN/{0}{1}.png".format(x, "." + str(0) + "." + str(ran_x1) + "." + str(ran_y1) +
                                                             "." + str(ran_x2) + "." + str(ran_y2)))

        # elif 40000 < int(num) <= 50000:
        #     background_resize.save(r"E:/AIdata/yellow_man2/TRAIN/{0}{1}.png".format(x, "." + str(1) + "." + str(0) + "." + str(0) +
        #                                                      "." + str(0) + "." + str(0)))
        elif 50000 < int(num) < 54000:
            background_resize.paste(img_new, (ran_x1, ran_y1), mask=a)

            ran_x2 = ran_x1 + ran_w
            ran_y2 = ran_y1 + ran_w
            background_resize.save(r"E:/AIdata/yellow_man2/TEST/{0}{1}.png".format(x, "." + str(0) + "." + str(ran_x1) + "." + str(ran_y1) +
                                                            "." + str(ran_x2) + "." + str(ran_y2)))
        else:
            background_resize.save(r"E:/AIdata/yellow_man2/TEST/{0}{1}.png".format(x, "." + str(1) + "." + str(0) + "." + str(0) +
                                                            "." + str(0) + "." + str(0)))


    except PIL.UnidentifiedImageError:
        continue
