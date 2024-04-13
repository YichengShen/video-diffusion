import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

plt.rcParams["savefig.bbox"] = 'tight'


def show(images):
    if not isinstance(images, list):
        raise Exception("images must be a list")
    fix, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
