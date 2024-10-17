import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from glob import glob
import cv2

"""
Compares YOLO box results of two images
"""
class L1BoxComparison:
    def __init__(self):        
        pass

    def prepare_coco_results(results):
        pass

    def compare_images(true_results, test_results):
        pass





"""
Plotting Utils
"""
class PlottingUtils:
    def load_images_from_folder(path):
        imgs = []
        for f in glob(f"{path}\\*.*"):
            img = cv2.imread(f)[..., ::-1]
            imgs.append(img)

        return imgs

    def plot_image_comparison(images, titles, figtitle = "", scale = 1.5, dpi = 200):
        width = len(images[0]) + 1
        height = len(images)

        ## get ratios
        total = 1
        width_ratios = [1]
        for img in images[0]:
            aspect = img.shape[1] / img.shape[0]
            width_ratios.append(aspect)
            total += aspect
        total /= width
        
        fig = plt.figure(figsize=(scale * width * total, scale * height * 0.9), dpi=dpi)
        fig.suptitle(figtitle)

        gs = gridspec.GridSpec(height, width, width_ratios=width_ratios)

        idx = 0
        for h in range(height):
            for w in range(width):
                ax = plt.subplot(gs[idx])
                ax.set_axis_off()
                
                if h == 0 and w > 0:
                    ax.set_title(f"({chr(ord('a') + idx - 1)})")

                if w == 0:
                    ax.text(0.5, 0.5, titles[h], horizontalalignment="center")
                else:
                    img = images[h][w - 1]
                    ax.imshow(img)

                idx += 1

        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        #fig.savefig("./compare/out.pdf")

        return fig