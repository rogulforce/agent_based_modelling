# Piotr Rogula, 249801
from matplotlib import pyplot as plt
from matplotlib import colors
from natsort import natsorted
import os
import imageio.v2 as imageio


class GifTool:
    """ tool visualizing the model."""
    def __init__(self, pic_dir: str = 'data/temp', gif_dir: str = 'data/gif', cmap=None, bounds=None):
        self.pic_dir = pic_dir
        self.gif_dir = gif_dir

        self.clear_dir(self.pic_dir)

        if not cmap:
            self.cmap = colors.ListedColormap(['white', 'red'])
            bounds = [-.5, .5, 1.5]
        else:
            self.cmap = cmap
            bounds = bounds

        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)
        self.img_num = 0

    def save_pic(self, data, title=None):
        self.visualize(data)
        if title:
            plt.title(title)
        plt.savefig(f'data/temp/fig{self.img_num}')
        plt.close()
        self.img_num += 1

    def visualize(self, data):
        plt.figure(figsize=(10, 10))
        img = plt.imshow(data, interpolation='nearest', origin='lower',
                         cmap=self.cmap, norm=self.norm)

        # return img

    def save_gif(self, name, **kwargs):
        images = []
        for file_name in natsorted(os.listdir(self.pic_dir)):
            if file_name.endswith('.png'):
                file_path = os.path.join(self.pic_dir, file_name)
                images.append(imageio.imread(file_path))

        imageio.mimsave(f'{self.gif_dir}/{name}', images, **kwargs)

    @staticmethod
    def clear_dir(directory):
        for file_name in os.listdir(directory):
            if file_name.endswith('.png'):
                os.remove(f'{directory}/{file_name}')
