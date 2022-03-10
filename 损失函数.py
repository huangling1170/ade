import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from torch.nn.functional import one_hot
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda"
train_data = datasets.CIFAR10(r"E:\AIdata\CIFAR10", train=True, transform=transforms.ToTensor(), download=False)
test_data = datasets.CIFAR10(r"E:\AIdata\CIFAR10", train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(train_data, batch_size=500, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.ReLU(),
            nn.Linear(80, 10),
            nn.LogSoftmax(dim=1)  #null 如果用softmax,没有错，损失精度都是正常的趋势，但是loss会出现负数，改成logsoftmax,优化速度更快
        )

    def forward(self, x):
        return self.fc_layer(x)

class net_v2(nn.Module):
    def __init__(self):
        super(net_v2, self).__init__()
        self.fayer = nn.Sequential(
            nn.Conv2d(3, 52, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(52, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256* 2 * 2, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        out_fayer = self.fayer(x)
        out_fayer = out_fayer.reshape(-1,256 * 2 * 2)
        return self.fc_layer(out_fayer)

if __name__ == '__main__':
    net = net_v2().to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    # loss_func = nn.MSELoss()
    loss_func = nn.NLLLoss()
    summaryWriter = SummaryWriter("../20211207MTCNN/logs")

    test_step = 0
    train_step = 0
    log_test_loss = 100  #随机给的数值
    for epoch in range(10000):
        # 加载训练损失
        sum_train_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            # imgs = imgs.reshape(-1, 32 * 32 * 3)      #卷积用了TOtensor就不用换形状了
            outs = net(imgs)
            # labels = one_hot(labels, 10).float()  #用null，标签不用做ont处理，并且标签必须为整形
            train_loss = loss_func(outs, labels)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            sum_train_loss = sum_train_loss + train_loss.cpu().item()#

            if i % 10 == 0 and i != 0:
                avg_train_loss = sum_train_loss/10
                print(f"epoch==>{epoch}", f"i===>{i}", f"avg_train_loss==={avg_train_loss}")
                summaryWriter.add_scalar("avg_train_loss", avg_train_loss, train_step)
                train_step += 1
                sum_train_loss = 0

        # 加载验证损失
        sum_acc = 0
        sum_test_loss = 0
        for i, (img, label) in enumerate(test_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            # img = img.reshape(-1, 32 * 32 * 3)
            out = net(img)
            # label = one_hot(label, 10).float()
            test_loss = loss_func(out, label)
            # score = torch.mean(torch.eq(torch.argmax(out,dim=1), torch.argmax(label,dim=1)).float())#
            score = torch.mean(torch.eq(torch.argmax(out, dim=1), label).float())  #null，并没有再做hot，直接使用
            sum_acc = sum_acc + score
            sum_test_loss = sum_test_loss + test_loss

            if i % 10 == 0 and i != 0:
                _score = sum_acc / 10
                avg_test_loss = sum_test_loss / 10
                # print(f"epoch==>{epoch}",f"i===>{i}",f"test_loss==={test_loss}")
                print(f"epoch==>{epoch}", f"i===>{i}", f"avg_test_loss==={avg_test_loss}")  # 平均损失会比较流畅
                print(f"epoch==>{epoch}", f"i===>{i}", f"_score==={_score}")
                summaryWriter.add_scalar("avg_test_loss", avg_test_loss, test_step)
                summaryWriter.add_scalar("_score", _score, test_step)
                test_step += 1
                sum_test_loss = 0
                _score= 0
                sum_acc = 0
                if avg_test_loss < log_test_loss:
                    torch.save(net.state_dict(), f"../20211207MTCNN/param/cifar.pt")
                    print("参数保存成功")
                    log_test_loss = avg_test_loss
            torch.save(net.state_dict(), f"param/{epoch}.pt")
            print("epoch参数保存成功")


