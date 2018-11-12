import ImgUtil
import numpy as np

imgs1 = ImgUtil.load_mnist(0)
cov1 = ImgUtil.image_to_cov(imgs1[0], 28)
cov2 = ImgUtil.image_to_cov(imgs1[1], 28)
dist = np.linalg.norm(cov1 - cov2)
print(dist)

imgs2 = ImgUtil.load_mnist(1)
cov3 = ImgUtil.image_to_cov(imgs2[0], 28)
cov4 = ImgUtil.image_to_cov(imgs2[1], 28)
dist = np.linalg.norm(cov3 - cov4)
print(dist)
print()
print(np.linalg.norm(cov2 - cov4))

