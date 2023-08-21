from PIL import Image
import torch.utils.data as data
from torchvision import transforms


class ImageDataSet(data.Dataset):
    def __init__(self, imgs, targets, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.imgs = imgs
        self.targets = targets

    def __getitem__(self, idx):
        img, target = self.imgs[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

        def __len__(self):
            return self.imgs.shape[0]


def dataset_from_labels(imgs, targets, class_set, **kwargs):
    class_mask = (targets[:, None] == class_set[None, :]).any(dim=1)
    return ImageDataSet(imgs=imgs[class_mask], targets=targets[class_mask], **kwargs)
