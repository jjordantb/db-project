import ImgUtil

imgs = ImgUtil.load_mnist(1)
img = imgs[4]

ImgUtil.draw_image(img, 28, 28)