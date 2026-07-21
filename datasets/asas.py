# from skimage import io
# import numpy as np
# from RS_HRSCD import Color2Index
# lab = io.imread("/root/autodl-fs/HRSCD/train/labelA/14-2012-0420-6895-LA93-0M50-E080_r0002_c0028.png")
# print(np.unique(Color2Index(lab)))

import numpy as np
from skimage import io

# 替换成你本地任意一张labelA路径
label_path = "/root/autodl-fs/HRSCD/train/labelA/14-2012-0420-6895-LA93-0M50-E080_r0002_c0028.png"
img = io.imread(label_path)
# 获取图片里所有独一无二的RGB颜色
unique_rgb = np.unique(img.reshape(-1, 3), axis=0)
print("图片中存在的所有RGB颜色：")
print(unique_rgb)