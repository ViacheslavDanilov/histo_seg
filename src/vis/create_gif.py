from glob import glob

import imageio

if __name__ == '__main__':
    images = []
    experiment_name = 'Experiment_00'
    for filename in sorted(
        glob(f'models/unet_se_resnet50_3101_2254/images_per_epoch/{experiment_name}_epoch_*.png'),
    ):
        images.append(imageio.imread(filename))
    imageio.mimsave(f'models/unet_se_resnet50_3101_2254/{experiment_name}.gif', images)
