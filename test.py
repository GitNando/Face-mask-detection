# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

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

def set_device():
    # check cuda availabilty
    if torch.cuda.is_available():
        dev = 0
    else:
        dev = 'cpu'
    print('to device: ', torch.device(dev))
    return torch.device(dev)

def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()
    best_acc = 0
    for epoch in range(n_epochs):
        print('Epoch number %d ' % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.00 * running_correct / total
        print('Training Dataset.\tGot %d out of %d images correctly (%.3f%%). Epoch loss: %.3f'
            %   (running_correct, total, epoch_acc, epoch_loss))
            
        test_dataset_acc = evaluate_model_on_test_set(model, test_loader)

        if(test_dataset_acc > best_acc):
            best_acc = test_dataset_acc
            save_checkpoint(model, epoch, optimizer, best_acc)

    print('Finished')
    return model

def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch+1,
        'model': model.state_dict(),
        'best accuracy': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, 'model_best_checkpoint.pth.tar')

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted==labels).sum().item()

    epoch_acc = 100.0 * predicted_correctly_on_epoch / total
    print('Testing Dataset.\tGot %d out of %d images correctly (%.3f%%)'
        % (predicted_correctly_on_epoch, total, epoch_acc))
    return epoch_acc

if __name__ == '__main__':
    # print(os.listdir('./Classification/imgs/train'))
    train_dataset_path = './Classification/imgs/train'
    test_dataset_path = './Classification/imgs/test'
    
    train_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)), 
                                        transforms.ToTensor(),
                                        ])

    test_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)), 
                                        transforms.ToTensor()
                                        ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    mean, std = get_mean_and_std(train_loader)
    
    train_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)), 
                                        transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(10),
                                        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                                        ])

    test_transforms = transforms.Compose([
                                        transforms.Resize((224, 224)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
                                        ])
    
    resnet18_model = models.resnet18(weights=None)
    num_features = resnet18_model.fc.in_features
    number_of_classes = 10
    resnet18_model.fc = nn.Linear(num_features, number_of_classes)
    device = set_device()
    resnet18_model = resnet18_model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

    train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 5)