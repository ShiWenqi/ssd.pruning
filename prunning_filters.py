from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import *
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd
from eval import test_net
from train import train_epoch

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
from layers import *

os.environ['CUDA_VISIBLE_DEVICES']='1'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()


def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]


def prune_conv_layer(net, layer_index, filter_index):
    # Bool: is_layer_from_vgg
    # Bool: is_feature_layer
    # confirm the boundary conditions with Hou Yunzhong


    # index of extras = index - index_bias
    index_bias = len(net.vgg)
    feature_layers = [21, 33, index_bias + 1, index_bias + 3,
                      index_bias + 5, index_bias + 7]

    if layer_index <= 34:
        is_layer_from_vgg = True
    else:
        is_layer_from_vgg = False

    if layer_index in feature_layers:
        is_feature_layer = True
    else:
        is_feature_layer = False

    if is_layer_from_vgg:
        _, conv = list(net.vgg._modules.items())[layer_index]

        # if layer_index == 33 then next_conv = 1st conv layer in extras
        if layer_index == 33:
            next_name, next_conv = list(net.extras._modules.items())[0]
        else:
            next_conv = None
            offset = 1
            while layer_index + offset < len(list(net.vgg._modules.items())):
                res = list(net.vgg._modules.items())[layer_index + offset]
                if isinstance(res[1], torch.nn.modules.conv.Conv2d):
                    next_name, next_conv = res
                    break
                offset = offset + 1
    else:
        _, conv = list(net.extras._modules.items())[layer_index-index_bias]
        next_conv = None
        offset = 1
        while layer_index - index_bias + offset < len(list(net.extras._modules.items())):
            res = list(net.extras._modules.items())[layer_index - index_bias + offset]
            if isinstance(res[1], torch.nn.modules.conv.Conv2d):
                next_name, next_conv = res
                break
            offset = offset + 1

    is_bias_present = False
    if conv.bias is not None:
        is_bias_present = True

    new_conv = \
        torch.nn.Conv2d(in_channels=conv.in_channels, \
                        out_channels=conv.out_channels - 1,
                        kernel_size=conv.kernel_size, \
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=is_bias_present)

    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
    new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    new_conv.weight.data = torch.from_numpy(new_weights).cuda()

    bias_numpy = conv.bias.data.cpu().numpy()

    bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
    bias[:filter_index] = bias_numpy[:filter_index]
    bias[filter_index:] = bias_numpy[filter_index + 1:]
    new_conv.bias.data = torch.from_numpy(bias).cuda()

    if next_conv is not None:
        is_bias_present = False
        if next_conv.bias is not None:
            is_bias_present = True
        next_new_conv = \
            torch.nn.Conv2d(in_channels=next_conv.in_channels - 1, \
                            out_channels=next_conv.out_channels, \
                            kernel_size=next_conv.kernel_size, \
                            stride=next_conv.stride,
                            padding=next_conv.padding,
                            dilation=next_conv.dilation,
                            groups=next_conv.groups,
                            bias=is_bias_present)

        old_weights = next_conv.weight.data.cpu().numpy()
        new_weights = next_new_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        next_new_conv.weight.data = torch.from_numpy(new_weights).cuda()

        next_new_conv.bias.data = next_conv.bias.data

    if is_layer_from_vgg and layer_index != 33:
        # front conv layer in vgg
        features = torch.nn.Sequential(
            *(replace_layers(net.vgg, i, [layer_index, layer_index + offset], \
                             [new_conv, next_new_conv]) for i, _ in enumerate(net.vgg)))
        del net.vgg
        del conv
        net.vgg = features
    elif layer_index == 33:
        # last layer in vgg
        features_vgg = torch.nn.Sequential(
            *(replace_layers(net.vgg, i, [layer_index], \
                             [new_conv]) for i, _ in enumerate(net.vgg)))
        features_extras = torch.nn.Sequential(
            *(replace_layers(net.extras, i, [0], \
                             [next_new_conv]) for i, _ in enumerate(net.extras)))
        del net.vgg
        del net.extras
        del conv
        net.vgg = features_vgg
        net.extras = features_extras
    elif next_conv is not None:
        # front layer in extras
        features = torch.nn.Sequential(
            *(replace_layers(net.extras, i, [layer_index - index_bias, layer_index + offset - index_bias], \
                             [new_conv, next_new_conv]) for i, _ in enumerate(net.extras)))
        del net.extras
        del conv
        net.extras = features
    else:
        # last layer in extras
        features_extras = torch.nn.Sequential(
            *(replace_layers(net.extras, i, [layer_index - index_bias], \
                             [new_conv]) for i, _ in enumerate(net.extras)))
        del net.extras
        del conv
        net.extras = features_extras


    if is_feature_layer:
        # is feature layer in SSD
        # modify loc and conf layers
        feature_layer_index = feature_layers.index(layer_index)
        _, loc_conv = list(net.loc._modules.items())[feature_layer_index]
        _, conf_conv = list(net.conf._modules.items())[feature_layer_index]

        is_bias_present = False
        if loc_conv.bias is not None:
            is_bias_present = True
        new_loc_conv = \
        torch.nn.Conv2d(in_channels=loc_conv.in_channels - 1, \
                        out_channels=loc_conv.out_channels, \
                        kernel_size=loc_conv.kernel_size, \
                        stride=loc_conv.stride,
                        padding=loc_conv.padding,
                        dilation=loc_conv.dilation,
                        groups=loc_conv.groups,
                        bias=is_bias_present)

        old_weights = loc_conv.weight.data.cpu().numpy()
        new_weights = new_loc_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        new_loc_conv.weight.data = torch.from_numpy(new_weights).cuda()

        new_loc_conv.bias.data = loc_conv.bias.data

        is_bias_present = False
        if conf_conv.bias is not None:
            is_bias_present = True
        new_conf_conv = \
        torch.nn.Conv2d(in_channels=conf_conv.in_channels - 1, \
                        out_channels=conf_conv.out_channels, \
                        kernel_size=conf_conv.kernel_size, \
                        stride=conf_conv.stride,
                        padding=conf_conv.padding,
                        dilation=conf_conv.dilation,
                        groups=conf_conv.groups,
                        bias=is_bias_present)

        old_weights = conf_conv.weight.data.cpu().numpy()
        new_weights = new_conf_conv.weight.data.cpu().numpy()

        new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]
        new_conf_conv.weight.data = torch.from_numpy(new_weights).cuda()

        new_conf_conv.bias.data = conf_conv.bias.data

        features_loc = torch.nn.Sequential(
            *(replace_layers(net.loc, i, [feature_layer_index], \
                             [new_loc_conv]) for i, _ in enumerate(net.loc)))
        del net.loc
        del loc_conv
        net.loc = features_loc

        features_conf = torch.nn.Sequential(
            *(replace_layers(net.conf, i, [feature_layer_index], \
                             [new_conf_conv]) for i, _ in enumerate(net.conf)))
        del net.conf
        del conf_conv
        net.conf = features_conf

        '''
        features = torch.nn.ModuleList(
            [replace_layers(net.vgg, i, [layer_index, layer_index + offset], \
                             [new_conv, next_new_conv]) for i, _ in enumerate(net.vgg)])

        '''
        # adjust L2Norm parameter
        _, conv21 = list(net.vgg._modules.items())[21]
        net.L2Norm = L2Norm(conv21.out_channels, 20)

    return net

if __name__ == '__main__':
    cfg = voc
    ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    net.load_weights(args.trained_model)
    net.eval()


    t0 = time.time()
    layer_index = 31
    #filter_index = 222

    for i in range(300):
        net = prune_conv_layer(net, layer_index, i)

    layer_index = 33
    for i in range(300):
        net = prune_conv_layer(net, layer_index, i)

    layer_index = 40
    for i in range(100):
        net = prune_conv_layer(net, layer_index, i)

    print("The prunning took", time.time() - t0)


    annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
    imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                              'Main', '{:s}.txt')
    YEAR = '2007'
    devkit_path = args.voc_root + 'VOC' + YEAR
    dataset_mean = (104, 117, 123)
    set_type = 'test'

    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(300, dataset_mean),
                           VOCAnnotationTransform())

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
