#20220712 HDR splicing tampering
# input:LDR+HDR
# output:Tampering+mask64
#Steps:
#1. Load image
#2. Image binarization
#3. Obtain and filter a connected domain
#4. Generate corresponding mask
#5. Do not overlap LDR and HDR repeatedly
#6. Output tampered image

import cv2
import numpy as np
from glob import glob

import tqdm
from matplotlib import pyplot as plt


class Config:
    # load_path
    ldr_path = './'
    hdr_path = './'
    # save_path
    img_path = './'
    mask_path = './'
    img_mask_path = './'

arge = Config()


def Dataset_IMG(img_paths):
    Img = np.asarray(glob(img_paths + '*.png'))
    Img = Img[::25]
    return Img


# ConnectedComponentsWithStats
def CCWStats(img, out, ldr):
    img = img.reshape(1060, 1900, 1).astype(np.uint8)
    # plt.imshow(img),plt.show()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    stats = stats[stats[:, 4].argsort()]
    for stat in stats:
        if stat[4] <= 64 * 64 * 3:
            labels[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]] = 0
            out[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]] = ldr[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]]
        else:
            break
    labels[labels > 0] = 1
    return labels, out

# main 
def TAM(ldr_path, hdr_path):
    # read
    ldr = cv2.imread(ldr_path)
    hdr = cv2.imread(hdr_path)
    img_gray = cv2.cvtColor(ldr, cv2.COLOR_BGR2GRAY)
    # threshold
    ret, binary = cv2.threshold(img_gray, 50, 220, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  #
    # plt.imshow(binary), plt.title('binary'), plt.show()
    # connectedComponentsWithStats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    stats = stats[stats[:, 4].argsort()][::-1]
    stats = stats[stats[:, 2] > 64]
    stats = stats[stats[:, 3] > 64]
    bboxs = stats[1, :4]  # y, x, h, w
    # Get connected domain location
    # a = np.where(labels[bboxs[1]:bboxs[1] + bboxs[3], bboxs[0]:bboxs[0] + bboxs[2]] > 0)
    # b1 = a[0] + bboxs[1]
    # b2 = a[1] + bboxs[0]
    # b = (b1, b2)
    # labels[b] = -1
    # labels[labels > -1] = 0
    # labels *= -1
    # plt.imshow(labels), plt.title('labels'), plt.show()
    # Tampering and mask generation
    k=0
    for i in bboxs:
        i = i//64
        bboxs[k]=i*64
        k+=1

    out = ldr.copy()
    img_h, img_w = labels.shape
    mask64 = np.zeros((img_h, img_w))
    mask64[bboxs[1]:bboxs[1] + bboxs[3], bboxs[0]:bboxs[0] + bboxs[2]] = 255
    out[bboxs[1]:bboxs[1] + bboxs[3], bboxs[0]:bboxs[0] + bboxs[2]] = hdr[bboxs[1]:bboxs[1] + bboxs[3],
                                                                      bboxs[0]:bboxs[0] + bboxs[2]]
    labels[labels>0] = 1
    for ih in range(bboxs[1], bboxs[1] + bboxs[3], 64):
        for iw in range(bboxs[0], bboxs[0] + bboxs[2], 64):
            # if ih + 64 > img_h:
            #     ih = img_h - 64
            # if iw + 64 > img_w:
            #     iw = img_w - 64
            if labels[ih:ih + 64, iw:iw + 64].sum() < 1024:
                # if mask64[ih:ih + 64, iw:iw + 64].sum() > 1024:
                mask64[ih:ih + 64, iw:iw + 64] = 0
                out[ih:ih + 64, iw:iw + 64, :] = ldr[ih:ih + 64, iw:iw + 64, :]
    mask64, out = CCWStats(mask64, out, ldr)
    mask64 = mask64 * 255
    # Tamper connected area mark > = tamper area
    mask_bboxs = Rectangle_IMG(bboxs, ldr)
    # plt.imshow(mask_bboxs), plt.title('out'), plt.show()
    return out, mask64, mask_bboxs

# Mark tamper area
def Rectangle_IMG(bboxs, BGR):
    color = (0, 0, 255)  # Red color in BGR；red：rgb(255,0,0)
    thickness = 10
    mask_BGR = BGR  # cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    b = bboxs
    x0, y0 = b[0], b[1]
    x1 = b[0] + b[2]
    y1 = b[1] + b[3]
    start_point, end_point = (x0, y0), (x1, y1)
    mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
    return mask_bboxs


def Save_tamper(img, mask, img_mask, number):
    cv2.imwrite(arge.img_path + str(number) + '.png', img)
    cv2.imwrite(arge.mask_path + str(number) + '.png', mask)
    cv2.imwrite(arge.img_mask_path + str(number) + '_sig.jpg', img_mask)


if __name__ == '__main__':
    print('--------------Start----------------')
    # Read file directory
    ldr_loder = Dataset_IMG(arge.ldr_path)
    hdr_loder = Dataset_IMG(arge.hdr_path)
    Total_num = ldr_loder.shape[0]
    Now_num = 0
    # Read file
    for ldr_path in tqdm.tqdm(ldr_loder):
        # Get the corresponding image
        hdr_path = hdr_loder[Now_num]
        # Tampering and mask generation
        Tampered_img, Tampered_mask, LDR_mask = TAM(ldr_path, hdr_path)
        Save_tamper(Tampered_img, Tampered_mask, LDR_mask, Now_num)
        Now_num += 1
