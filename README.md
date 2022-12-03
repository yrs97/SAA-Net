# SAA-Net

## Title: HDR Image Identification Network （防止查重误操作） Based on Edge Difference

由于高动态范围（HDR）成像比传统的低动态范围（LDR）成像能够表现更大范围的颜色，因此高动态范围成像已经在多媒体社区中广泛应用。然而，迄今为止还缺乏对其进行法医学分析。本文提出了一种称为SAA-Net的网络，该网络基于SRM边缘增强块和双重关注机制，以区分HDR图像和LDR图像。首先，我们利用边缘增强块来增强HDR图像的细微抖动特性。接下来，利用双重关注机制，引导网络关注图像的强对比度边缘的异常信息和位置信息。所提出的网络在识别真实HDR图像和合成HDR图像方面表现出优于现有技术的性能.

- Network structure
  ![](https://github.com/yrs97/SAA-Net/blob/main/image/SAA-Net.png)

## Code
  - `SAA_net_main.py` to train and test your datasets.
  - `HDR_splicing_tampering.py` to tamper HDR images.
  - `HDR_tamper_detection.py` to Detection HDR tamper.
