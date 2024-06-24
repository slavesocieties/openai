from skimage.restoration import denoise_tv_chambolle
from preprocess import *

def anisotropic_diffusion(image, weight=0.1):
    return denoise_tv_chambolle(image, weight=weight)

blocked_image = preprocess_image('images/239746-0088.jpg')
im = Image.fromarray(blocked_image)
im.show()
diffused_image = anisotropic_diffusion(blocked_image)
im = Image.fromarray(blocked_image)
im.show()
