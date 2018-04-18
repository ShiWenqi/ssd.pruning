# Add by Shi Wenqi
# implement functions: get_candidates_to_prune(),
#                      prune(),


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

os.environ['CUDA_VISIBLE_DEVICES']='2'

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


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

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


def get_candidates_to_prune(num_filters_to_prune, net):

    # "self.prunner.reset()"
    net.reset()

    train_epoch(net, epoch_num=1, rank_filters=True)

    # "self.prunner.normalize_ranks_per_layer()"
    net.normalize_ranks_per_layer()

    prune_targets = net.get_prunning_plan(num_filters_to_prune)

    return prune_targets



def prune(net, args):
    # get the accuracy before prunning

    '''
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)

    '''
    # “self.model.train()” ??

    number_of_filters = net.total_num_filters()
    num_filters_to_prune_per_iteration = 512
    iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
    iterations = int(iterations * 2.0 / 3)

    print(("Number of prunning iterations to reduce 67% filters", iterations))

    # structure of "prune_targets"?
    iterations = 1
    for _ in range(iterations):
        print("Ranking filters..")
        prune_targets = get_candidates_to_prune(num_filters_to_prune_per_iteration, net)

        '''
        layers_prunned = {}
        for layer_index, filter_index in prune_targets:
            if layer_index not in layers_prunned:
                layers_prunned[layer_index] = 0
            layers_prunned[layer_index] = layers_prunned[layer_index] + 1

        print(("Layers that will be prunned", layers_prunned))
        print("Prunning filters..")
        net = net.cpu()
        for layer_index, filter_index in prune_target:
            net = prune_conv_layer(net, layer_index, filter_index)

        net = net.cuda()
        message = ste(100*float(net.total_num_filters()) / number_of_filters) + "%"
        print(("Filter prunned", str(message)))

        # "self.test()"???
        print("Fine tuning to recover from prunning iteration.")
        net = train_epoch(net, epoch_num=10, rank_filters=False)


    print("Finished. Going to fine tune the model a bit more")
    net = train_epoch(net, epoch_num=15, rank_filters=False)
    torch.save(net, "model prunned")

'''


if __name__ == '__main__':
    cfg = voc
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    ssd_net.load_weights(args.trained_model)

    if args.cuda:
        net = net.cuda()

    prune(net, args)