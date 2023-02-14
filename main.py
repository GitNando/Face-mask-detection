import torch
from torchvision import models

alexnet = models.alexnet(pretrained=True)
alexnet

def get_mean_and_std(loader):
    mean = 0.
    std = 0. 
    total_images_count = 0
    for images, _ in loader:
        images_count_in_a_batch = images.size(0)
        # print(images.shape)
        images = images.view(images_count_in_a_batch, images.size(1), -1)
        # print(images.shape)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images_count_in_a_batch
    mean /= total_images_count
    std /= total_images_count

    return mean, std


if __name__ == '__main__':
    train_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    ])
    mean, std = get_mean_and_std(train_loader)