import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
import torch
from torch import nn, optim
from PIL import Image
from collections import OrderedDict
import time

from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg19', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model, criterion, optimizer, data_loaders, epochs, gpu):
    steps = 0
    print_every = 10
    for epoch in range(epochs):
        running_loss = 0
        for il, (inputs, labels) in enumerate(data_loaders[0]): 
            steps += 1 

            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda') 
                device = torch.device("cuda")
            else:
                model.cpu()
                
            optimizer.zero_grad()
            
            # Forward and backward passes
            result = model.forward(inputs)
            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy=0

                for il, (inputs,labels) in enumerate(data_loaders[1]): 
                        optimizer.zero_grad()
                        
                        if gpu == 'gpu':
                            inputs, labels = inputs.to('cuda') , labels.to('cuda') 
                            model.to('cuda:0') 
                        else:
                            pass 
                        
                        with torch.no_grad():    
                            result= model.forward(inputs)
                            val_loss = criterion(result,labels)
                            probability = torch.exp(result).data
                            equal = (labels.data == probability.max(1)[1])
                            accuracy += equal.type_as(torch.FloatTensor()).mean()

                val_loss= val_loss/len(data_loaders[1])
                accuracy= accuracy /len(data_loaders[1])
                
                print(f'Epoch:{epochs+1}/{epochs}'
                    f'Running Loss: {running_loss/print_every:.4f}'
                    f'Validation Loss: {val_loss}'
                    f'Accuracy: {accuracy:.4f}')

                running_loss = 0
            
def main():
    print("Todo OK?")  
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    

    train_transforms = transforms.Compose([transforms.Resize(255), 
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255), 
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255), 
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    image_datasets = [datasets.ImageFolder(train_dir, transform=train_transforms), 
                  datasets.ImageFolder(valid_dir, transform=valid_transforms),
                  datasets.ImageFolder(test_dir, transform=test_transforms)]

    data_loaders = [torch.utils.data.DataLoader(image_datasets[0],batch_size=64,shuffle=True),
              torch.utils.data.DataLoader(image_datasets[1],batch_size=64,shuffle=True),
              torch.utils.data.DataLoader(image_datasets[2],batch_size=64,shuffle=True)]
   

    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    if args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    
    elif args.arch == "vgg19":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))


    model.classifier = classifier
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    gpu = args.gpu 
    train(model, criterion, optimizer, data_loaders, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir 
    save_checkpoint(path, model, optimizer, args, classifier)


if __name__ == "__main__":
    main()
