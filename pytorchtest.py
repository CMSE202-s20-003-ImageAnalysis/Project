from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

def run():
    torch.multiprocessing.freeze_support()
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data__dir'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def imshow(inp, title=None):
        """Imshow for Tensor.""" 
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.savefig("transferlearningwithout.jpg")  # pause a bit so that plots are updated


    # # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))

    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=24)
    visualize_model(model_ft)

if __name__ == '__main__':
    run()

## Without Fractals
# Epoch 0/23
# ----------
# train Loss: 0.8077 Acc: 0.5161
# val Loss: 1.3680 Acc: 0.4340

# Epoch 1/23
# ----------
# train Loss: 0.8714 Acc: 0.4871
# val Loss: 1.0640 Acc: 0.4057

# Epoch 2/23
# ----------
# train Loss: 0.9174 Acc: 0.5323
# val Loss: 1.1678 Acc: 0.4434

# Epoch 3/23
# ----------
# train Loss: 1.0291 Acc: 0.4968
# val Loss: 1.0445 Acc: 0.3868

# Epoch 4/23
# ----------
# train Loss: 0.8853 Acc: 0.5355
# val Loss: 0.8793 Acc: 0.4906

# Epoch 5/23
# ----------
# train Loss: 0.9093 Acc: 0.4677
# val Loss: 0.9282 Acc: 0.4906

# Epoch 6/23
# ----------
# train Loss: 0.8530 Acc: 0.5065
# val Loss: 1.0873 Acc: 0.4245

# Epoch 7/23
# ----------
# train Loss: 0.7166 Acc: 0.5613
# val Loss: 0.8471 Acc: 0.4434

# Epoch 8/23
# ----------
# train Loss: 0.6899 Acc: 0.5871
# val Loss: 0.8031 Acc: 0.4811

# Epoch 9/23
# ----------
# train Loss: 0.6869 Acc: 0.5710
# val Loss: 0.9005 Acc: 0.4245

# Epoch 10/23
# ----------
# train Loss: 0.6942 Acc: 0.5419
# val Loss: 0.8106 Acc: 0.4811

# Epoch 11/23
# ----------
# train Loss: 0.6930 Acc: 0.5839
# val Loss: 0.8042 Acc: 0.5000

# Epoch 12/23
# ----------
# train Loss: 0.6721 Acc: 0.5677
# val Loss: 0.8061 Acc: 0.4340

# Epoch 13/23
# ----------
# train Loss: 0.6817 Acc: 0.5710
# val Loss: 0.8452 Acc: 0.4151

# Epoch 14/23
# ----------
# train Loss: 0.6666 Acc: 0.6097
# val Loss: 0.8435 Acc: 0.3962

# Epoch 15/23
# ----------
# train Loss: 0.6649 Acc: 0.5806
# val Loss: 0.8175 Acc: 0.4434

# Epoch 16/23
# ----------
# train Loss: 0.6653 Acc: 0.5903
# val Loss: 0.8328 Acc: 0.4528

# Epoch 17/23
# ----------
# train Loss: 0.6464 Acc: 0.5903
# val Loss: 0.8253 Acc: 0.4151

# Epoch 18/23
# ----------
# train Loss: 0.6688 Acc: 0.5968
# val Loss: 0.8198 Acc: 0.4623

# Epoch 19/23
# ----------
# train Loss: 0.6435 Acc: 0.6226
# val Loss: 0.8170 Acc: 0.4057

# Epoch 20/23
# ----------
# train Loss: 0.6365 Acc: 0.6323
# val Loss: 0.8214 Acc: 0.4528

# Epoch 21/23
# ----------
# train Loss: 0.6503 Acc: 0.6355
# val Loss: 0.8193 Acc: 0.4245

# Epoch 22/23
# ----------
# train Loss: 0.6511 Acc: 0.6226
# val Loss: 0.8228 Acc: 0.3962

# Epoch 23/23
# ----------
# train Loss: 0.6693 Acc: 0.5871
# val Loss: 0.8198 Acc: 0.4151

# Training complete in 7m 4s
# Best val Acc: 0.500000

## With
# Epoch 0/23
# ----------
# train Loss: 0.9003 Acc: 0.5430
# val Loss: 1.1050 Acc: 0.5000

# Epoch 1/23
# ----------
# train Loss: 0.8917 Acc: 0.4967
# val Loss: 1.0293 Acc: 0.5526

# Epoch 2/23
# ----------
# train Loss: 0.8222 Acc: 0.5596
# val Loss: 0.9785 Acc: 0.5000

# Epoch 3/23
# ----------
# train Loss: 0.8173 Acc: 0.4934
# val Loss: 0.7292 Acc: 0.4737

# Epoch 4/23
# ----------
# train Loss: 1.0375 Acc: 0.5166
# val Loss: 2.5448 Acc: 0.5088

# Epoch 5/23
# ----------
# train Loss: 1.0239 Acc: 0.5166
# val Loss: 1.1332 Acc: 0.4912

# Epoch 6/23
# ----------
# train Loss: 0.8347 Acc: 0.5298
# val Loss: 0.8243 Acc: 0.5175

# Epoch 7/23
# ----------
# train Loss: 0.7016 Acc: 0.5530
# val Loss: 0.7330 Acc: 0.5175

# Epoch 8/23
# ----------
# train Loss: 0.7235 Acc: 0.5397
# val Loss: 0.7473 Acc: 0.5614

# Epoch 9/23
# ----------
# train Loss: 0.7122 Acc: 0.5265
# val Loss: 0.7492 Acc: 0.5000

# Epoch 10/23
# ----------
# train Loss: 0.7048 Acc: 0.5397
# val Loss: 0.7376 Acc: 0.4649

# Epoch 11/23
# ----------
# train Loss: 0.6866 Acc: 0.5530
# val Loss: 0.7462 Acc: 0.5351

# Epoch 12/23
# ----------
# train Loss: 0.7028 Acc: 0.5364
# val Loss: 0.7162 Acc: 0.4912

# Epoch 13/23
# ----------
# train Loss: 0.7218 Acc: 0.5000
# val Loss: 0.7449 Acc: 0.5263

# Epoch 14/23
# ----------
# train Loss: 0.6877 Acc: 0.5629
# val Loss: 0.7269 Acc: 0.5263

# Epoch 15/23
# ----------
# train Loss: 0.6868 Acc: 0.5828
# val Loss: 0.7111 Acc: 0.5175

# Epoch 16/23
# ----------
# train Loss: 0.6985 Acc: 0.5430
# val Loss: 0.7219 Acc: 0.4825

# Epoch 17/23
# ----------
# train Loss: 0.6958 Acc: 0.5298
# val Loss: 0.7092 Acc: 0.5088

# Epoch 18/23
# ----------
# train Loss: 0.6890 Acc: 0.5662
# val Loss: 0.7164 Acc: 0.5000

# Epoch 19/23
# ----------
# train Loss: 0.6803 Acc: 0.6060
# val Loss: 0.7243 Acc: 0.5263

# Epoch 20/23
# ----------
# train Loss: 0.6909 Acc: 0.5629
# val Loss: 0.7062 Acc: 0.5175

# Epoch 21/23
# ----------
# train Loss: 0.6734 Acc: 0.5695
# val Loss: 0.7168 Acc: 0.5000

# Epoch 22/23
# ----------
# train Loss: 0.6773 Acc: 0.5795
# val Loss: 0.7143 Acc: 0.5000

# Epoch 23/23
# ----------
# train Loss: 0.6881 Acc: 0.5695
# val Loss: 0.7289 Acc: 0.5263

# Training complete in 2m 56s
# Best val Acc: 0.561404