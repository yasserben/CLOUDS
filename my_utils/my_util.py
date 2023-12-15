import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import torch
from PIL import Image


def colorize_mask(mask, add=None):
    palette = [
        128,
        64,
        128,
        244,
        35,
        232,
        70,
        70,
        70,
        102,
        102,
        156,
        190,
        153,
        153,
        153,
        153,
        153,
        250,
        170,
        30,
        220,
        220,
        0,
        107,
        142,
        35,
        152,
        251,
        152,
        70,
        130,
        180,
        220,
        20,
        60,
        255,
        0,
        0,
        0,
        0,
        142,
        0,
        0,
        70,
        0,
        60,
        100,
        0,
        80,
        100,
        0,
        0,
        230,
        119,
        11,
        32,
        0,
        0,
        0,
    ]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
        # mask: numpy array of the mask
    if add is not None:
        add = add.cpu()
        add = torch.squeeze(add)
        mask[add == 0] = 19
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
    new_mask.putpalette(palette)
    return new_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return "{:.2f} seconds".format(self.avg)


def generate_indices(max_index, percentage, seed=0):
    """
    Generate indices for training and validation set and save them to a file
    Args:
        max_index: the maximum index of the dataset
        percentage: the percentage of the dataset to be used for training/validation
        seed: the seed for the random number generator

    Returns: the indices for the training/validation set

    """
    np.random.seed(seed)
    indices = np.array([str(x).zfill(5) + ".png" for x in range(max_index)])
    np.random.shuffle(indices)
    np.savetxt(
        f"./dataset_creation/gta5_style_{int(percentage*100)}.txt",
        indices[: int(max_index * percentage)],
        fmt="%s",
    )
    return indices[: int(max_index * percentage)]


def create_source_dataset(
    num_samples, max_index=24966, root="/home/ids/benigmim/dataset", seed=0
):
    """
    Create a dataset of the source images with num_images images
    """
    np.random.seed(seed)
    indices = np.array([str(x).zfill(5) + ".png" for x in range(max_index)])
    np.random.shuffle(indices)
    create_directory(os.path.join(root, f"gta5_{num_samples}_samples"))
    for i in indices[:num_samples]:
        shutil.copy(
            os.path.join(root, "gta5/images", i),
            os.path.join(root, f"gta5_{num_samples}_samples", i),
        )


def create_directory(dir):
    """
    Create a directory if it does not exist
    Args:
        dir: the directory to be created

    Returns: None

    """
    try:
        os.makedirs(dir, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' can not be created")


def generate_mosaic(num_images, filename, filepath, size):
    """
    Generate a mosaic of the images in the filepath using torchvision
    """
    import torchvision
    import glob
    import os
    from PIL import Image
    from torchvision import transforms

    # convert_tensor = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    convert_tensor = transforms.Compose(
        [transforms.Resize((640, 1088)), transforms.Resize(size), transforms.ToTensor()]
    )

    # Get all images
    # images = list(filename.read().splitlines())
    with open(filename) as f:
        images = f.read().splitlines()
    # images = glob.glob(filename)

    # Create a list of images
    imgs = []
    for i in images[:num_images]:
        imgs.append(convert_tensor(Image.open(os.path.join(filepath, i))))
        # imgs.append(Image.open(os.path.join(filepath,i)))

    # imgs[-2] =imgs[-2][:, :, :-3]

    # Create a new image of the same size
    mosaic = torchvision.utils.make_grid(imgs, nrow=1, padding=1)

    # Save the mosaic
    torchvision.utils.save_image(
        mosaic, os.path.join(filepath, f"mosaic_{num_images}.png")
    )


def generate_list_ckpt(training_steps, checkpoint_steps):
    """
    Generate a list of checkpoints using the maximum number of training_steps
    Args:
        training_steps: the maximum number of training steps
        checkpoint_steps: the number of steps between each checkpoint

    Returns:

    """
    return [x for x in range(checkpoint_steps, training_steps + 1, checkpoint_steps)]


def visualize_semantic_map(pred, mask=None):
    pred = pred.unsqueeze(dim=0)
    pred = torch.argmax(pred, dim=1)
    pred_1 = torch.squeeze(pred)
    pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, mask)
    plt.imshow(pred_1_map)
    plt.show()
    return pred_1_map


def save_semantic_map_maxed(pred, after=False):
    pred_1 = torch.squeeze(pred)
    pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, None)
    im = pred_1_map
    if after:
        im.save("./semantic_map_after.png")
    else:
        im.save("./semantic_map_before.png")
    return pred_1_map


def visualize_semantic_map_maxed(pred, mask=None):
    pred_1 = torch.squeeze(pred)
    pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, mask)
    plt.imshow(pred_1_map)
    plt.show()
    return pred_1_map


def get_rgb_from_semantic_map_maxed(pred, mask=None):
    pred_1 = np.asarray(pred, dtype=np.uint8)
    pred_1_map = colorize_mask(pred_1, mask)
    return pred_1_map


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def visualize_seg(image, means, stds, i):
    image = torch.clamp(denorm(image, means, stds), 0, 1)
    image = image[i]
    image = image.cpu().detach().numpy()
    plt.imshow(image.transpose(1, 2, 0))
    plt.show()


def visualize_rgb(image, i=0):
    image = image[i]
    image = image.cpu().detach().numpy()
    image = image.transpose((1, 2, 0))
    plt.imshow(image)
    plt.show()

def save_rgb(image, i=0):
    image = image[i]
    image = image.cpu().detach().numpy()
    image = image.transpose((1, 2, 0))
    im = Image.fromarray(image)
    im.save("rgb_image_gen.jpg")


def visualize_mask(mask, i=0):
    # Check shape of mask
    if len(mask.shape) == 3:
        # If mask is a torch tensor
        if isinstance(mask, torch.Tensor):
            mask = mask[i]
            mask = mask.cpu().detach().numpy()
            plt.imshow(mask, cmap="gray")
            plt.show()
        # If mask is a numpy array
        else:
            plt.imshow(mask[i], cmap="gray")
            plt.show()
    elif len(mask.shape) == 2:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().detach().numpy()
            plt.imshow(mask, cmap="gray")
            plt.show()
        else:
            plt.imshow(mask, cmap="gray")
            plt.show()
