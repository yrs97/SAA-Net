#20220714 HDR tamper detection
# input: img, mask
# output: score

import cv2
import glob
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from matplotlib import pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Config:
    img_path = './'
    mask_path = './'
    pkl_path = './'
    save_path = './'


arge = Config()

# load images
def Load_file():
    Imgs = np.asarray(glob.glob(arge.img_path + '*.png'))
    Masks = np.asarray(glob.glob(arge.mask_path + '*.png'))
    return Imgs, Masks

def Read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = img / 255.
    return torch.from_numpy(img).float()

# load Net
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


#Fuse_mask_iou
def Fuse_mask_iou(Pre_mask, Mask):
    # plt.imshow(Pre_mask),plt.title('Pre'),plt.show()
    # plt.imshow(Mask), plt.title('Org'), plt.show()
    Pre_mask = CCWStats(Pre_mask)
    fuse_img = Pre_mask+Mask
    _H,_W = fuse_img.shape
    W_img = np.zeros((_H,_W,1))
    R_img = np.zeros((_H,_W,1))
    B_img = np.zeros((_H,_W,1))
    out_img = np.zeros((_H,_W,3))

    #                   Positives = HDR,    Negatives = LDR
    #  True  =  HDR   :       White               Red
    #  False =  LDR   :       Blue                Black

    # White
    W_img[fuse_img == 2] = 255
    W = np.sum(W_img)/255.

    # Red
    R_img[(W_img + Pre_mask.reshape(_H,_W,1))==1] = 255
    R = np.sum(R_img)/255.

    #Blue
    B_img[(W_img + Mask.reshape(_H,_W,1))==1] = 255
    B = np.sum(B_img)/255.

    #out_img
    out_img[:, :, :] = W_img
    out_img[:, :, 0:1] += R_img
    out_img[:, :, 2:3] += B_img

    #Score
    HDR_Precision = W/(W+R)
    HDR_Recall = W/(W+B)
    F1 = 2*HDR_Precision*HDR_Recall / (HDR_Precision + HDR_Recall)


    return out_img, HDR_Precision, HDR_Recall, F1

#ConnectedComponentsWithStats
def CCWStats(img):
    img = img.reshape(1060, 1900, 1).astype(np.uint8)
    # plt.imshow(img),plt.show()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    stats = stats[stats[:, 4].argsort()]
    for stat in stats:
        if stat[4] <= 64*64*4:
            labels[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]] = 0
        # elif stat[4] > 64*64*3 and stat[4] < 64*64*8:
        #     labels[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]] = 1
        else:
            break
    # stats = stats[stats[:, 4] > 16384]
    labels[labels > 0] = 1
    return labels





if __name__ == '__main__':
    Imgs, Masks = Load_file()

    Net = New_cadm().to(DEVICE)
    model_path = arge.pkl_path + 'model_best.pth'
    print(model_path)
    Net.load_state_dict(torch.load(model_path))

    HDR_Precision = 0
    HDR_Recall = 0
    F1 = 0
    L_ = len(Imgs)
    Num = 0
    for Num in tqdm.tqdm(range(0,L_)):
        Img = Read_img(Imgs[Num]).to(DEVICE)
        C, H, W = Img.shape
        Pre_mask = np.zeros((H,W))
        for ih in range(0, H, 64):
            for iw in range(0, W, 64):
                # if ih + 64 > H:
                #     ih = H-64
                # if iw + 64 > W:
                #     iw = W-64

                # temp_ = torch.tensor(Img[:, ih: ih + 64, iw: iw + 64]).unsqueeze(0).to(DEVICE)
                temp_ = Img[:, ih: ih + 64, iw: iw + 64].unsqueeze(0)
                pre_label = Net(temp_)
                pre_out = pre_label.argmax(dim=-1)

                if pre_out == 0:
                    Pre_mask[ih: ih + 64, iw: iw + 64] = 0
                else:
                    Pre_mask[ih: ih + 64, iw: iw + 64] = 1
        Mask = cv2.imread(Masks[Num])
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY) / 255.
        Pre_mask_iou, HDR_Precision_, HDR_Recall_, F1_ = Fuse_mask_iou(Pre_mask, Mask)
        HDR_Precision += HDR_Precision_
        HDR_Recall += HDR_Recall_
        F1 += F1_

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(Pre_mask_iou/255.)
        # ax.set_title('Score:'+str(Score)[:6])
        # plt.savefig(arge.save_path+str(Num)+'.png')
        # plt.show()

        No_num = Masks[Num].split('\\')[1].split('.')[0]
        print(No_num)
        plt.imshow(Pre_mask_iou/255.)
        plt.title('F1:'+str(F1_)[:6] + '   HDR_Precision   '+str(HDR_Precision_)[:6]+'   HDR_Recall   '+str(HDR_Recall_)[:6])
        plt.savefig(arge.save_path + No_num + '.png')
        plt.show()
    HDR_Precision /= L_
    HDR_Recall /= L_
    F1 /= L_
    print(f'HDR_Precision:{HDR_Precision}\nHDR_Recall:{HDR_Recall}\nF1:{F1}')

