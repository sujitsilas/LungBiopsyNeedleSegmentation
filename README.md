# LungBiopsyNeedleSegmentation
Lung Biopsy Needle Segmentation in CT-Scans

Segmentation has grown to be an essential component in a wide number of applications ranging from improving vehicle self-driving capabilities of vehicles to facilitating the process of medical image analysis. It involves the portioning of images into multiple segments in order to distinguish between regions of interest and oth-er regions [1]. Medical image segmentation is a critical component in a clinical setting that can help with diagnosis, treatment plan-ning, and overtime disease monitoring [2]. The goal of this study is to segment lung biopsy needles in CT-scans. We use a U-Net archi-tecture pretrained VGG16 backbone to perform binary segmentation of biopsy needles in CT-scans. Our model achieves a precision of 0.5819 and a composite score of 0.5322 calculated as a the weighted mean of the dice score and sensitivity. 


Keywords. Segmentation, U-Net, Needle Segmentation, CT Scans

1	Minaee, S., Boykov, Y., Porikli, F., Plaza, A., Kehtarnavaz, N., & Terzopoulos, D. (2022). Image Segmentation Using Deep Learning: A Survey. IEEE Transactions on Pattern Analysis and Machine Intelli-gence, 44(7), 3523–3542. https://doi.org/10.1109/TPAMI.2021.3059968

2	Ma, J., He, Y., Li, F., Han, L., You, C., & Wang, B. (2024). Segment anything in medical images. Nature Communications, 15(1), 654. https://doi.org/10.1038/s41467-024-44824-z
![image](https://github.com/sujitsilas/LungBiopsyNeedleSegmentation/assets/70657722/a4b42030-5bdf-489d-a631-7ce571e70869)

