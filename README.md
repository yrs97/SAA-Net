# SAA-Net

## Title: HDR Image Identification Network Based on Edge Difference

Due to its ability to represent a greater range of colors than conventional low dynamic range (LDR) imaging, high dynamic range (HDR) imaging has been widely applied in the multimedia community. However, to our best knowledge, forensic studies tailored for the authenticity recognition of HDR images are lacking. This paper proposes a network called SAA-Net, which aims to differentiate HDR images from LDR ones, based on an SRM edge enhancement block and the dual attention mechanism. First, we leverage the edge enhancement block to enhance the subtle jitter characteristics of HDR images. Next, with the double attention mechanism, the network is guided to pay attention to the abnormal information and location information of the strong contrast edge of the image. The proposed network shows superior performance over the state-of-the-art on identifying both real HDR images and synthetic HDR images.

- Network structure
  ![](https://github.com/yrs97/SAA-Net/blob/main/image/SAA-Net.png)

## Code
  - `SAA_net_main.py` to train and test your datasets.
  - `HDR_splicing_tampering.py` to tamper HDR images.
  - `HDR_tamper_detection.py` to Detection HDR tamper.
