# LungBiopsyNeedleSegmentation


Lung Biopsy Needle Segmentation in CT Scans
Segmentation has become an essential component in a wide range of applications, from enhancing the self-driving capabilities of vehicles to facilitating medical image analysis. It involves partitioning images into multiple segments to distinguish between regions of interest and other regions [1]. Medical image segmentation is a critical component in a clinical setting that can aid in diagnosis, treatment planning, and disease monitoring over time [2].

The goal of this study is to segment lung biopsy needles in CT scans. We utilize a U-Net architecture with a pretrained VGG16 backbone to perform binary segmentation of biopsy needles in CT scans. Our model achieves a precision of 0.5819 and a composite score of 0.5322, calculated as the weighted mean of the dice score and sensitivity.

Keywords
Segmentation, U-Net, Needle Segmentation, CT Scans

References
Minaee, S., Boykov, Y., Porikli, F., Plaza, A., Kehtarnavaz, N., & Terzopoulos, D. (2022). Image Segmentation Using Deep Learning: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7), 3523–3542. https://doi.org/10.1109/TPAMI.2021.3059968

Ma, J., He, Y., Li, F., Han, L., You, C., & Wang, B. (2024). Segment anything in medical images. Nature Communications, 15(1), 654. https://doi.org/10.1038/s41467-024-44824-z
