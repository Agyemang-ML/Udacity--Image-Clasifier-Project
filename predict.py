#importing libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

#define a parser for augument scripts
parser = argparse.ArgumentParser(description="Parser of prediction script")
parser.add_argument(
    'image_dir', help='Provide path to image. Mandatory argument', type=str)
parser.add_argument(
    'load_dir', help='Provide path to checkpoint. Mandatory argument', type=str)
parser.add_argument(
    '--top_k', help='Top K most likely classes. Optional', type=int)
parser.add_argument('--category_names',
                    help='Mapping of categories to real names. JSON file name to be provided. Optional', type=str)
parser.add_argument('--GPU', help="Option to use GPU. Optional", type=str)


# load of model and building it
def loading_model(file_path):
    checkpoint = torch.load(file_path)
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']

    for param in model.parameters():
        param.requires_grad = False  # turning off tuning of the model

    return model

#functions to get PIL image to use in Pytorch


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)  # loading image
    width, height = im.size  # original size

    # smallest part: width or height should be kept not more than 256
    if width > height:
        height = 256
        img.thumbnail((50000, height), Image.ANTIALIAS)
    else:
        width = 256
        img.thumbnail((width, 50000), Image.ANTIALIAS)

    width, height = img.size
    #crop 224x224 in the center
    cut = 224
    left = (width - cut)/2
    top = (height - cut)/2
    right = left + 224
    down = up + 224
    img = img.crop((left, up, right, down))

    np_image = np.array(img)/255  # to make values from 0 to 1
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])
    np_image = np_image.transpose((2, 0, 1))

    return np_image

#function to predict, prediction function


def predict(image_path, model, topkl, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)  # loading image and processing it
    if device == 'cuda':
        img = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        img = torch.from_numpy(image).type(torch.FloatTensor)

    # used to make size of torch as expected. as forward method is working with batches,
    img = img.unsqueeze(dim=0)
    #doing that we will have batch size = 1

    #enabling GPU/CPU
    model.to(device)
    img.to(device)

    with torch.no_grad():
        output = model.forward(img)
    output_prob = torch.exp(output)  # converting into a probability

    probs, indices = output_prob.topk(topkl)
    probs = probs.cpu().numpy()  # converting both to numpy array
    indeces = indeces.cpu().numpy()

    probs = probs.tolist()[0]  # converting both to list
    indices = indices.tolist()[0]

    mapping = {val: key for key, val in
               model.class_to_idx.items()
               }

    classes = [mapping[item] for item in indices]
    classes = np.array(classes)  # converting to Numpy array

    return probs, classes


#arguments to accept for data loading
args = parser.parse_args()
file_path = args.image_dir

#condition for either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#loading JSON file
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

#loading model
model = loading_model(args.load_dir)

if args.top_k:
    nm_cl = args.top_k
else:
    nm_cl = 1

#calculating probabilities and classes
probs, classes = predict(file_path, model, nm_cl, device)

#preparing class_names using mapping with cat_to_name
class_names = [cat_to_name[item] for item in classes]

for l in range(nm_cl):
    print("Number: {}/{}.. ".format(l+1, nm_cl),
          "Class name: {}.. ".format(class_names[l]),
          "Probability: {:.3f}..% ".format(probs[l]*100),
          )
