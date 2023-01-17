#!/usr/bin/env python3

import torch

import sys
import os
import pandas as pd
import numpy as np
import time

from PIL import Image

from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.models as models

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import cm

from enum import Enum

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML,CSS
import base64
from io import BytesIO

import cv2

from preprocess.preprocess import preprocess

class Models(Enum):
    resnet18 = 'resnet18'
    alexnet = 'alexnet'
    squeezenet = 'squeezenet'
    vgg16 = 'vgg16'
    densenet = 'densenet'
    inception = 'inception'
    googlenet = 'googlenet'
    shufflenet = 'shufflenet'
    mobilenet_v2 = 'mobilenet_v2'
    mobilenet_v3_large = 'mobilenet_v3_large'
    mobilenet_v3_small = 'mobilenet_v3_small'
    resnext50_32x4d = 'resnext50_32x4d'
    wide_resnet50_2 = 'wide_resnet50_2'
    mnasnet = 'mnasnet'
    efficientnet_b0 = 'efficientnet_b0'
    efficientnet_b1 = 'efficientnet_b1'
    efficientnet_b2 = 'efficientnet_b2'
    efficientnet_b3 = 'efficientnet_b3'
    efficientnet_b4 = 'efficientnet_b4'
    efficientnet_b5 = 'efficientnet_b5'
    efficientnet_b6 = 'efficientnet_b6'
    efficientnet_b7 = 'efficientnet_b7'
    regnet_y_400mf = 'regnet_y_400mf'
    regnet_y_800mf = 'regnet_y_800mf'
    regnet_y_1_6gf = 'regnet_y_1_6gf'
    regnet_y_3_2gf = 'regnet_y_3_2gf'
    regnet_y_8gf = 'regnet_y_8gf'
    regnet_y_16gf = 'regnet_y_16gf'
    regnet_y_32gf = 'regnet_y_32gf'
    regnet_x_400mf = 'regnet_x_400mf'
    regnet_x_800mf = 'regnet_x_800mf'
    regnet_x_1_6gf = 'regnet_x_1_6gf'
    regnet_x_3_2gf = 'regnet_x_3_2gf'
    regnet_x_8gf = 'regnet_x_8gf'
    regnet_x_16gf = 'regnet_x_16gf'
    regnet_x_32gf = 'regnet_x_32gf'

    def __str__(self):
        return self.value


class Classifier(nn.Module):
    def __init__(self, model, classes):
        super(Classifier, self).__init__()

        if model == Models.resnet18:
            self.model = models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.alexnet:
            self.model = models.alexnet(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096,classes)

        elif model == Models.squeezenet:
            self.model = models.squeezenet1_0(pretrained=True)
            self.model.classifier[1] = nn.Conv2d(512, classes, kernel_size=(1,1), stride=(1,1))

        elif model == Models.vgg16:
            self.model = models.vgg16(pretrained=True)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs,classes)

        elif model == Models.densenet:
            self.model = models.densenet161(pretrained=True)
            self.model.classifier = nn.Linear(1024, classes)

        elif model == Models.inception:
            self.model = models.inception_v3(pretrained=True)
            self.model.AuxLogits.fc = nn.Linear(768, classes)
            self.model.fc = nn.Linear(2048, classes)

        elif model == Models.googlenet:
            self.model = models.googlenet(pretrained=True)
            self.model.fc = nn.Linear(1024, classes)

        elif model == Models.shufflenet:
            self.model = models.shufflenet_v2_x1_0(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.mobilenet_v2:
            self.model = models.mobilenet_v2(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.mobilenet_v3_large:
            self.model = models.mobilenet_v3_large(pretrained=True)
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, classes)

        elif model == Models.mobilenet_v3_small:
            self.model = models.mobilenet_v3_small(pretrained=True)
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, classes)

        elif model == Models.resnext50_32x4d:
            self.model = models.resnext50_32x4d(pretrained=True)
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, classes)

        elif model == Models.wide_resnet50_2:
            self.model = models.wide_resnet50_2(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.mnasnet:
            self.model = models.mnasnet1_0(pretrained=True)
            self.model.classifier[1] = nn.Linear(1280, classes)

        elif model == Models.efficientnet_b0:
            self.model = models.efficientnet_b0(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.efficientnet_b1:
            self.model = models.efficientnet_b1(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.efficientnet_b2:
            self.model = models.efficientnet_b2(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.efficientnet_b3:
            self.model = models.efficientnet_b3(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.efficientnet_b4:
            self.model = models.efficientnet_b4(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.efficientnet_b5:
            self.model = models.efficientnet_b5(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.efficientnet_b6:
            self.model = models.efficientnet_b6(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.efficientnet_b7:
            self.model = models.efficientnet_b7(pretrained=True)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_y_400mf:
            self.model = models.regnet_y_400mf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_y_800mf:
            self.model = models.regnet_y_800mf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_y_1_6gf:
            self.model = models.regnet_y_1_6gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_y_3_2gf:
            self.model = models.regnet_y_3_2gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_y_8gf:
            self.model = models.regnet_y_8gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_y_16gf:
            self.model = models.regnet_y_16gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_y_32gf:
            self.model = models.regnet_y_32gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_x_400mf:
            self.model = models.regnet_x_400mf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_x_800mf:
            self.model = models.regnet_x_800mf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_x_1_6gf:
            self.model = models.regnet_x_1_6gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_x_3_2gf:
            self.model = models.regnet_x_3_2gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_x_8gf:
            self.model = models.regnet_x_8gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_x_16gf:
            self.model = models.regnet_x_16gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)

        elif model == Models.regnet_x_32gf:
            self.model = models.regnet_x_32gf(pretrained=True)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, classes)



    def forward(self, image):
        output = self.model(image)
        return output

class Hook():
    def __init__(self):
        self.stored = []

    def hook_func(self, m, i, o):
        self.stored.append(o.data.cpu().numpy())


class ResnetCAM(nn.Module):
    def __init__(self, path):
        super(ResnetCAM, self).__init__()

        self.resnet = torch.load(path)
        self.hook_output = Hook()
        self.resnet.model.layer4.register_forward_hook(self.hook_output.hook_func )

    def activations_hook(self):
        return self.hook_output.stored

    def forward(self, x):
        x = self.resnet(x)
        return x

    def weight_softmax(self):
        params = list(self.resnet.parameters())
        weight_softmax = np.squeeze(params[-2].data.numpy())
        return weight_softmax

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


model_paths = {

    'nodule': 'nodule-resnet18.pth',
    'pseudonodule': 'pseudonodule-resnet18.pth',
    'lung_metastasis': 'lung_metastasis-resnet18.pth',
    'bone_metastasis': 'bone_metastasis-resnet18.pth',
    'normal': 'normal-resnet18.pth',
}
model_flder_path = 'models'

models = {}

for k,v in model_paths.items():
    print(k)
    models[k] = ResnetCAM(os.path.join(model_flder_path, v))
    models[k].cpu()
    models[k].eval()

classes = {
            0: 'control',
            1: 'positive'
        }

transform = transforms.Compose([
                            transforms.Resize((1024,1024)),
                            transforms.ToTensor()
                            ])

img_path = sys.argv[1]



image = preprocess(img_path, "MONOCHROME2")
image = image[...,::-1]
img_trans = transform(Image.fromarray(image)).unsqueeze(0)
img_trans.shape


results = {}

for k,model in models.items():
    r = {}
    r['out'] = model(img_trans)

    h_x = F.softmax(r['out'], dim=1).data.squeeze()
    #print(h_x)
    #probs, idx = h_x.sort(0, True)
    #probs = probs.numpy()
    #idx = idx.numpy()
    r['probs'] = h_x.numpy()
    r['idx'] = [0,1]

    results[k] = r



df_results = pd.DataFrame([{'finding': k, 'prob': v['probs'][1]} for k,v in results.items()])


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (1024, 1024)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        #cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


irgb = np.array(image)
i = np.stack((irgb[:,:,2],)*3, axis=-1)

pil_img = Image.fromarray(i)
buff = BytesIO()
pil_img.save(buff, format="JPEG")
ib64 = base64.b64encode(buff.getvalue()).decode("utf-8")

pil_img = Image.fromarray(irgb)
buff = BytesIO()
pil_img.save(buff, format="JPEG")
irgb_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

#plt.imshow(i)


heatmaps = []
for k,v in results.items():
    CAMs = returnCAM(models[k].activations_hook()[0], models[k].weight_softmax(), [v['idx'][1]])
    height, width, _ = i.shape
    img_num = cv2.resize(i, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC) #img_trans[0].permute(1, 2, 0).numpy()
    img_num = img_num/255.
    heatmap = CAMs[0]
    colormap = cm.get_cmap('inferno')
    heatmap_col = colormap(heatmap)
    heatmap = heatmap-0.2
    heatmap[heatmap > 0.5] = 0.5
    heatmap = np.stack((heatmap,)*3, axis=-1)
    #heatmap[heatmap>255] = 255
    heatmap[heatmap<0] = 0
    result = heatmap_col[:,:,:-1]*heatmap + img_num*(1-heatmap)
    print(heatmap_col.max(),heatmap_col.min())
    #cv2.imwrite('CAM.jpg', result)

    result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_CUBIC)  #img_trans[0].permute(1, 2, 0).numpy()

    pil_img = Image.fromarray((result * 255).astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    heatmap_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    heatmaps.append({'name': k, 'image': heatmap_b64})
    #fig = plt.figure(figsize=(10, 10))
    #fig.suptitle(f'{k} {classes[v["idx"][1]]} -> prob={v["probs"][1]}', fontsize=20)
    #plt.imshow(result)


logos_base64='''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
xR7IJCpX8wAAAABJRU5ErkJggg=='''

env = Environment(loader=FileSystemLoader('./templates/'))
template = env.get_template("main.html")


results_prob = [{'name':k, 'prob':"{:0>2.2%}".format(v['probs'][1]),'prob_val':v['probs'][1]}
                for k,v in results.items()
                if k!='normal']
results_zipped = zip(results_prob[::2],results_prob[1::2])



template_vars = {"title" : "Report",
                 'logos': logos_base64,
                 'original_image': ib64,
                 'cropped_image': irgb_b64,
                 'normal_prob': "{:0>2.2%}".format(results['normal']['probs'][0]),
                 'normal_prob_clamp': min(int(results['normal']['probs'][0]*10000)/100,98.8),
                 "results_table": results_zipped,#df_results.to_html(index=False),
                 'heatmaps': zip(heatmaps[::2],heatmaps[1::2])}

html_out = template.render(template_vars)
with open('html.html', 'w') as out:
    out.write(html_out)

HTML(string=html_out).write_pdf("report.pdf",stylesheets=[CSS('templates/w3.css')])

