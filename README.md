# SAA-Net

## HDR Image Identification Network Based on Edge Difference

### Due to its ability to represent a greater range of colors than conventional low dynamic range (LDR) imaging, high dynamic range (HDR) imaging has been widely applied in the multimedia community. However, the forensic analysis of it is lacking so far. This paper proposes a network called SAA-Net, which is based on an SRM edge enhancement block and the dual attention mechanism, to differentiate HDR images from LDR ones. First, we leverage the edge enhancement block to enhance the subtle jitter characteristics of HDR images. Next, with the double attention mechanism, the network is guided to pay attention to the abnormal information and location information of the strong contrast edge of the image. The proposed network shows superior performance over the state-of-the-art on identifying both real HDR images and synthetic HDR images.

# Code
  - 'SAA_net_main.py' to train and test your datasets.
  - 'HDR_tampering.py' to tamper HDR images.
  - 'HDR_tamper_detection.py' to Detection HDR tamper.
