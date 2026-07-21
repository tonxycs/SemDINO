# from skimage import io
# import numpy as np
# from RS_HRSCD import Color2Index
# lab = io.imread("~/HRSCD/train/labelA/14-2012-0420-6895-LA93-0M50-E080_r0002_c0028.png")
# print(np.unique(Color2Index(lab)))

import numpy as np
from skimage import io


label_path = "~/HRSCD/train/labelA/14-2012-0420-6895-LA93-0M50-E080_r0002_c0028.png"
img = io.imread(label_path)

unique_rgb = np.unique(img.reshape(-1, 3), axis=0)

print(unique_rgb)
