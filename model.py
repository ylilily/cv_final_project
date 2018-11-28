# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

# TODO: INITIALIZE DATALOADERS HERE
# Credit: http://blog.outcome.io/pytorch-quick-start-classifying-an-image/ (for loading ImageNet classes and images)
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
"""

########################################################################
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

if __name__ == '__main__':

# hardcoded input height/width parameters (as recommended by S. Iizuka et. al. paper)

    h = int(224)
    w = int(224)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Define the Convolution Neural Networks for colorization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            """
            Composing the layers for a sequential network
            Max-pooling is not used as such a pooling layer will distort the image
            Instead, we use a stride of 2 to improve info density (see: https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/)
            Intiutively, the conv filter "moves" through the image in 2-pixel steps instead of 1
            
            Also, unlike Keras, there is no padding=same option for the filters and colorization requires us to keep the image size the same
            Therefore, we use the formula padding = (kernel_size - stride)//2
            See discussion on https://github.com/pytorch/pytorch/issues/3867
            """
            #h = 224
            layers = []

            layers.append(nn.Conv2d(int(h/8), int(h/4), 3, stride=2, padding=(3 - 2)//2))
            layers.append(nn.Conv2d(int(h/4), int(h/2), 3, stride=1, padding=(3 - 1)//2))

            layers.append(nn.Conv2d(int(h/4), int(h/2), 3, stride=2, padding=(3 - 2)//2))
            layers.append(nn.Conv2d(int(h/2), int(h), 3, stride=1, padding=(3 - 1)//2))

            layers.append(nn.Conv2d(int(h/2), int(h), 3, stride=2, padding=(3 - 2)//2))
            layers.append(nn.Conv2d(int(h), int(h*2), 3, stride=1, padding=(3 - 1) // 2))

            #low-level features network
            self.low_features = nn.Sequential(*layers)

            global_layers = []
            global_layers.append(nn.Conv2d(int(h*2), int(h*2), 3, stride=2, padding=(3 - 2)//2))
            global_layers.append(nn.Conv2d(int(h*2), int(h*2), 3, stride=1, padding=(3 - 1) // 2))
            global_layers.append(nn.Conv2d(int(h * 2), int(h * 2), 3, stride=2, padding=(3 - 2) // 2))
            global_layers.append(nn.Conv2d(int(h * 2), int(h * 2), 3, stride=1, padding=(3 - 1) // 2))
            global_layers.append(nn.Linear(int(h*2), int(h*4)))
            global_layers.append(nn.Linear(int(h*4), int(h*2)))
            global_layers.append(nn.Linear(int(h*2), int(h)))

            self.global_features = nn.Sequential(*global_layers)

            mid_features_layers = []
            mid_features_layers.append(nn.Conv2d(int(h * 2), int(h * 2), 3, stride=1, padding=(3 - 1) // 2))
            mid_features_layers.append(nn.Conv2d(int(h), int(h * 2), 3, stride=1, padding=(3 - 1) // 2))

            self.mid_features = nn.Sequential(*mid_features_layers)

            color_layers1 = []
            color_layers1.append(nn.Conv2d(int(h/2), int(h), 3, stride=1, padding=(3 - 1) // 2))
            self.color1 = nn.Sequential(*color_layers1)

            color_layers2 = []
            color_layers2.append(nn.Conv2d(int(h/4), int(h/2), 3, stride=1, padding=(3 - 1) // 2))
            color_layers2.append(nn.Conv2d(int(h / 4), int(h/4), 3, stride=1, padding=(3 - 1) // 2))
            self.color2 = nn.Sequential(*color_layers2)

            color_layers3 = []
            color_layers3.append(nn.Conv2d(int(h/8), int(h/4), 3, stride=1, padding=(3 - 1) // 2))
            color_layers3.append(nn.Conv2d(2, int(h/8), 3, stride=1, padding=(3 - 1) // 2))
            self.color3 = nn.Sequential(*color_layers3)

        def forward(self, img):
            img = self.low_features(img)
            global_img = img
            global_img = self.global_features(global_img)
            img = self.mid_features(img)
            combined = torch.cat((img.view(img.size(0), -1),global_img.view(global_img.size(0), -1)), dim=1)
            combined = self.color1(combined)
            combined = F.upsample(combined, scale_factor=2, mode='nearest')
            combined = self.color2(combined)
            combined = F.upsample(combined, scale_factor=2, mode='nearest')
            combined = self.color3(combined)

            return combined

    net = Net()

# TODO: DEFINE LOSS FUNCTION HERE
########################################################################
# Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#TRAIN THE NETWORK
########################################################################
# Training the network
# ^^^^^^^^^^^^^^^^^^^^
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')