import os.path
import numpy as np
import torch.optim
from PIL import Image,ImageDraw
from yellow_data import yellow_data
from yellow_net import Net_v1, Net_v2
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2
from utils import iou
train_dataset = yellow_data("E:\AIdata\yellow_man2", is_train=True)
train_loder = DataLoader(train_dataset, batch_size=80, shuffle=True)
test_dataset = yellow_data("E:\AIdata\yellow_man2", is_train=False)
test_loder = DataLoader(test_dataset, batch_size=1, shuffle=True)

DEVICE = "cuda"
sava_path = r"E:\PycharmProjects\20211129my_yellow\model\net_v2.pt"

if __name__ == '__main__':
    # net = Net_v1().to(DEVICE)
    net = Net_v2().to(DEVICE)
    if os.path.exists(sava_path):
        net.load_state_dict(torch.load(sava_path))
        print("====已加载预训练权重====")

    loss_func = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters())
    train_sum_loss = 0
    test_sum_loss = 0

    Train = False
    for epoch in range(100000):
        if Train:
            for i, (x, y) in enumerate(train_loder):  # torch.Size([50, 300, 300, 3])  torch.Size([50, 5])
                # x = x.reshape(-1, 300 * 300 * 3).to(DEVICE)  # n,v  torch.Size([50, 270000])
                x = x.permute(0,3,1,2).to(DEVICE)
                y = y.to(DEVICE)

                out = net(x)
                train_loss = loss_func(out, y)

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                train_sum_loss = train_sum_loss + train_loss.cpu().item()

                if i % 10 == 0 and i != 0:
                    avg_train_loss = train_sum_loss / 10
                    print(f"epoch=={epoch}", f"i=={i}", f"avg_train_loss=={avg_train_loss}")
                    torch.save(net.state_dict(),sava_path)   #仅在训练集使用?
                    train_sum_loss = 0
            # del train_loss
        else:

            for i, (x, y) in enumerate(test_loder):
                # x = x.reshape(-1, 300 * 300 * 3).to(DEVICE)
                x = x.permute(0, 3, 1, 2).to(DEVICE)
                y = y.to(DEVICE)

                out = net(x)
                test_loss = loss_func(out, y)
                test_sum_loss = test_sum_loss + test_loss.cpu().item()

                if i % 10 == 0 and i != 0:
                      avg_test_loss = test_sum_loss / 10

                      print(f"epoch=={epoch}", f"i=={i}", f"avg_test_loss=={avg_test_loss}")
                      train_sum_loss = 0

                x = x.permute(0,2,3,1).cpu()
                out = out[0][1:5].detach().cpu().numpy()*300   #pil读图片是numpy格式
                y = y[0][1:5].detach().cpu().numpy()*300
                print("iou===",iou(out,y))
                img_data = np.array((x[0]+0.5)*255,dtype = np.uint8)   #RGB
                img_data = cv2.cvtColor(img_data,cv2.COLOR_RGB2BGR)
                cv2.rectangle(img_data,(int(y[0]),int(y[1])),(int(y[2]),int(y[3])),color=(0,0,255),thickness=2)
                cv2.rectangle(img_data, (int(out[0]), int(out[1])), (int(out[2]),int(out[3])), color=(0,255,255), thickness=2)
                cv2.imshow("yellow_man",img_data)
                cv2.waitKey()
                cv2.destroyAllWindows()
                # img = Image.fromarray(img_data,"RGB")
                # draw= ImageDraw.Draw(img)
                # draw.rectangle(np.array(y),outline="red",width=2)
                # draw.rectangle(np.array(out),outline="yellow",width=2)
                # img.show()



