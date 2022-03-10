import  os
from torch.utils.data import  Dataset,DataLoader
import numpy as np
import torch
from PIL import Image,ImageDraw
import cv2


class yellow_data(Dataset):
    def __init__(self,path,is_train = True):
        sub_dir = "TRAIN" if is_train else "TEST"
        self.path = path+"/"+sub_dir
        self.dataset = os.listdir(self.path)   #os读出来是列表,所以后面索引[]


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        label =torch.Tensor(np.array(self.dataset[index].split(".")[1:6],dtype =np.float32)/300) #最好是转成张量
        img_path = os.path.join(self.path,self.dataset[index])

        img = Image.open(img_path)  #用cv读会报错,hwc
        # print(type(img))
        # exit()
        # img = cv2.imread(img_path)
        # img_data = torch.Tensor(img/255-0.5)

        img_data =torch.Tensor(np.array(img)/255.-0.5)   #去均值化，0点对称
        print(img_data.shape)
        print(label.shape)
        exit()
        return img_data,label

if __name__ == '__main__':
    dataset= yellow_data(r"E:\AIdata\yellow_man2",is_train=True)
    dataloder =DataLoader(dataset,batch_size=50,shuffle=True)

    for i ,(imgs,labels) in enumerate(dataloder):
        # print(imgs.shape)     # 50,300,300,3
        # print(labels.shape)   # 50,5
        print(imgs[0])
        x = imgs[0].numpy()     #索引降维，hwc，刚好是np读图片的类型
        y = labels[0,1:5].numpy()  #标签是5个值，有一个判断，只取后面四个标签


        imgs_data  = np.array((x+0.5)*255,dtype = np.uint8)

        # img_data=cv2.rectangle(imgs_data,(int(y[0]*300),int(y[1]*300)),(int(y[2]*300),int(y[3]*300)),(255,0,0))
        # cv2.imshow("img",imgs_data)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        imgs = Image.fromarray(imgs_data,"RGB")
        print(imgs)
        exit()
        draw= ImageDraw.Draw(imgs)
        draw.rectangle(y*300,outline="red")
        imgs.show()







