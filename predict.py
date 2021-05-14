import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from PIL import Image
from math import ceil
import json
import os
import random
from utils import load_checkpoint, load_cat_names
from train import check_gpu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/3/image_06634.jpg') # use a deafault filepath to a primrose image 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):

    img_pil = Image.open(image) 
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    image = image_transforms(img_pil)
    
    return image

def predict(image_path, model, topk=3, gpu='gpu'):

    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
        
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1) 
    
    probs = np.array(probability.topk(topk)[0][0])

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [index_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

def print_result(probs, flowers):
    for a, b in enumerate(zip(flowers, probs)):
    print (f'Rank {(a+1)}: '
           f'Flower: {b[1]}, probability: {ceil(a[0]*100)}%')
    
def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    
    image_tensor = process_image(args.image)
    device = check_gpu(gpu_arg=args.gpu);
    top_probs, top_labels, top_flowers = predict(image_tensor, model,device, cat_to_name,args.top_k)
    
    print_result(top_flowers, top_probs)
    
if __name__ == "__main__":
    main()
