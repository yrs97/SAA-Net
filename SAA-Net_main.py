
import glob
import random
import torch
import tqdm
import torch.nn as nn
import numpy as np
from torch.nn import init
import torch.nn.functional as F
import cv2.cv2 as cv2
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.nn.parameter import Parameter
from PIL import Image
from torchvision import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Preset parameters
Config = {
    'gt_path': './',
    'med_path': './',
    'New': './',
    'img_size': 64,
    'in_channel': 3,
    'block_channel': 25,
    'batch_size': 8,
    'lr': 1e-5,
    'epoch': 50,
    'epoch_test': 1
}


# DataLoader
def load_file():
    LDR = np.asarray(glob.glob(Config['med_path'] + '*.png'))
    HDR = np.asarray(glob.glob(Config['gt_path'] + '*.png'))

    LDR = LDR[::19]
    HDR = HDR[::19]

    LDR1 = LDR[5000:]
    HDR1 = HDR[5000:]

    LDR = LDR[:5000]
    HDR = HDR[:5000]

    LDR_label = np.array([0] * len(LDR))
    HDR_label = np.array([1] * len(HDR))

    LDR_label1 = np.array([0] * len(LDR1))
    HDR_label1 = np.array([1] * len(HDR1))

    IMG = np.concatenate([LDR, HDR])
    LABEL = np.concatenate([LDR_label, HDR_label])

    IMG1 = np.concatenate([LDR1, HDR1])
    LABEL1 = np.concatenate([LDR_label1, HDR_label1])

    index = [i for i in range(len(IMG[0:]))]
    random.shuffle(index)
    train_img = np.asarray(IMG[index[:int(len(index) * 0.8)]])
    train_label = np.asarray(LABEL[index[:int(len(index) * 0.8)]])
    val_img = np.asarray(IMG[index[int(len(index) * 0.8):]])
    val_label = np.asarray(LABEL[index[int(len(index) * 0.8):]])

    train_loader = DataLoader(
        Ali_Dataset(img=train_img, label=train_label,
                    ),
        batch_size=Config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        Ali_Dataset(img=val_img, label=val_label,
                    ),
        batch_size=Config['batch_size'],
        shuffle=True
    )

    Test_loader = DataLoader(
        Ali_Dataset(img=IMG1, label=LABEL1,
                    ),
        batch_size=1,
        shuffle=False
    )

    return train_loader, val_loader, Test_loader


class Ali_Dataset(Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def read_cv2(self, img_path):
      # Read image and convert
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        return img


    def __getitem__(self, index):
        images = self.read_cv2(self.img[index]) / 255.
        label = self.label[index]
        return torch.from_numpy(images).float(), label

    def __len__(self):
        return len(self.img)




# Network=====================================

SRM_npy1 = np.load('SRM3_3.npy')

class pre_Layer_3_3(nn.Module):
    def __init__(self, stride=1, padding=1):
        super(pre_Layer_3_3, self).__init__()
        self.in_channels = 3
        self.out_channels = 25
        self.kernel_size = (3, 3)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(25, 3, 3, 3), requires_grad=True)
        self.bias = Parameter(torch.Tensor(25), requires_grad=True)

        self.reset_parameters()
        self.Sig = nn.Sigmoid()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy1
        self.bias.data.zero_()

    def forward(self, input):
        return self.Sig(F.conv2d(input, self.weight, self.bias, self.stride, self.padding))


class SEAttention(nn.Module):
    """ Attention"""

    def __init__(self, in_ch, reduction=8):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class New_cadm(nn.Module):
    def __init__(self):
        super(New_cadm, self).__init__()

        self.SRM = pre_Layer_3_3()
        self.conv1 = self.conv_layer(25, 64)
        self.conv2 = self.conv_layer(64, 128)
        self.conv3 = self.conv_layer(128,256)
        self.conv4 = self.conv_layer(256,512)

        self.SAM = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(25, 25, kernel_size=(3, 3), padding=(1, 1)),
            nn.Sigmoid())

        self.CADM1 = self.cadm_layer(25, 64)
        self.CADM2 = self.cadm_layer(64, 128)
        self.CADM3 = self.cadm_layer(128, 256)

        self.fc1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.Dropout(0.8),
            nn.Linear(1024, 512),
            nn.Dropout(0.8),
            nn.Linear(512, 2),
            nn.Softmax()

        )
    def cadm_layer(self, inc, outc):
        cadm = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.ReLU(),
            SEAttention(outc),
            nn.ReLU(),
        )
        return cadm

    def conv_layer(selg, inc, outc ):
        conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(outc, outc, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return conv



    def forward(self, x):
        x_3 = torch.log(x+1e-6)   #[8, 3, 128, 128]
        x_25 = self.SRM(x_3)         #[8, 25, 128, 128]
        x_25_ = self.SAM(x_25)       #[8, 25, 128, 128]
        x_25 = x_25 * x_25_

        x_64 = self.conv1(x_25)       #[8, 64, 64, 64]
        x_64_ = self.CADM1(x_25_)
        x_64 = x_64 * x_64_

        x_128 = self.conv2(x_64)       #[8, 128, 32, 32]
        x_128_ = self.CADM2(x_64_)
        x_128 = x_128 * x_128_

        x_256 = self.conv3(x_128)       #[8, 256, 16, 16]
        x_256_ = self.CADM3(x_128_)
        x_256 = x_256 * x_256_

        x = self.conv4(x_256)       #[8, 512, 8, 8]
        out = self.fc1(x)

        return out





# Train
def train_block(model, loader, optimizer):
    train_loss = []
    train_accs = []
    for i, (img, label) in enumerate(tqdm.tqdm(loader)):
        img = img.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        pre_label = model(img)  
        loss = loss_fn(pre_label, label.long())
        loss.backward()
        optimizer.step()

        acc = (pre_label.argmax(dim=-1) == label).float().mean()
        train_accs.append(acc.item())
        train_loss.append(loss.item())

    return np.array(train_accs).mean(), np.array(train_loss).mean()


@torch.no_grad()
def val_block(model, loader):
    losses = []
    accs = []
    model.eval()
    for image, label in tqdm.tqdm(loader):
        image, label = image.to(DEVICE), label.long().to(DEVICE)
        pre_label = model(image)  # .squeeze(1)
        loss = loss_fn(pre_label, label)
        acc = (pre_label.argmax(dim=-1) == label).float().mean()
        accs.append(acc.item())
        losses.append(loss.item())

    return np.array(accs).mean(), np.array(losses).mean()


def Train_main(model, train_iter, val_iter, optimizer):
    best_loss = 100.
    best_acc = 0.
    plot_train_acc = []
    plot_val_acc = []
    plot_train_loss = []
    plot_val_loss = []

    for epoch in range(1, Config['epoch'] + 1):
        print(f"epoch:{epoch}")
        train_acc, train_loss = train_block(model, train_iter, optimizer)
        val_acc, val_loss = val_block(model, val_iter)

        scheduler.step(val_loss)
        plot_train_acc.append(train_acc)
        plot_val_acc.append(val_acc)
        plot_train_loss.append(train_loss)
        plot_val_loss.append(val_loss)
        print(f'\ntrain_acc:{train_acc}')
        print(f'val_acc:{val_acc}')
        print(f'\ntrain_loss:{train_loss}')
        print(f'val_loss:{val_loss}')

        if val_loss < best_loss:
            best_loss = val_loss
            model_name = Config[DISPLAY] + 'model_best.pth'  #
            torch.save(model.state_dict(), model_name)
            print(f'model_name:{model_name}')

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % 10 == 0:
            # plot acc and loss
            plt.plot(plot_train_acc, ":", label="train_acc")
            plt.plot(plot_val_acc, ":", label="val_acc")
            plt.plot(plot_train_loss, label="train_loss")
            plt.plot(plot_val_loss, label="val_loss")
            title_ = DISPLAY + '-' + str(epoch)
            plt.title(title_)
            plt.legend()
            fig_ = Config[DISPLAY] + title_+'.png'
            plt.savefig(fig_)
            plt.show()

    print(f"best_acc: {best_acc}")
    print(f"best_loss: {best_loss}")


# val

@torch.no_grad()
def test_block(model, loader):
    losses = []
    accs = []
    model.eval()

    target_category = 1  # tabby, tabby cat
    for image, label in tqdm.tqdm(loader):
        image, label = image.to(DEVICE), label.long().to(DEVICE)
        pre_label = model(image)  # .squeeze(1)
        loss = loss_fn(pre_label, label)
        acc = (pre_label.argmax(dim=-1) == label).float().mean()
        accs.append(acc.item())
        losses.append(loss.item())

    return np.array(accs).mean(), np.array(losses).mean()


def Test_main(model, test_iter):
    plot_test_acc = []
    plot_test_loss = []

    for epoch in range(1, Config['epoch_test'] + 1):
        print(f"epoch:{epoch}")
        plot_test_acc, plot_test_loss = test_block(model, test_iter)

    print(f"plot_test_acc: {plot_test_acc}")
    print(f"plot_test_loss: {plot_test_loss}")


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)


DISPLAY = 'New'
model = New_cadm().to(DEVICE)
model.apply(weights_init_kaiming)
if __name__ == '__main__':
    print(f'Device:{DEVICE}')
    Train_loader, Val_loader, Test_loader = load_file()
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=Config['lr'],
                                  weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    Train_main(model, Train_loader, Val_loader, optimizer)
    print(f'\nTrain_Uet_End')

    # --------------------------------
    #Prediction
    pre_model = New_cadm().to(DEVICE)
    pre_model.load_state_dict(torch.load(Config[DISPLAY] + 'model_best.pth'))
    Test_main(pre_model, Test_loader)
    print(f'\nTest_Uet_End')

# ----------------------------------


