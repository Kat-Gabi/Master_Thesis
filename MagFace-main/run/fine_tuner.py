#!/usr/bin/env python

# Initial necessary imports
import sys
import os
# Specify the desired directory
desired_directory = '../MagFace-main' 

# Change the current working directory

print("PATHHHH","/".join(os.path.realpath(__file__).split("/")[0:-2]) )
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]) + "/MagFace-main")

sys.path.append("..")
from dataloader import dataloader
from models import magface
from utils import utils
import numpy as np
from collections import OrderedDict
from termcolor import cprint
from torchvision import datasets
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import argparse
import random
import warnings
import time
import pprint
import os
import wandb
from codecarbon import EmissionsTracker


warnings.filterwarnings("ignore")



# parse the args
cprint('=> parse the args ...', 'green')
parser = argparse.ArgumentParser(description='Trainer for Magface')
parser.add_argument('--pretrained', default='', type=str, help='Path to pretrained model')
parser.add_argument('--cpu_mode', default='1', type=str, help='1 for CPU or 0 for GPU')
parser.add_argument('--arch', default='iresnet18', type=str,
                    help='backbone architechture')
parser.add_argument('--train_list', default='', type=str,
                    help='')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--last-fc-size', default=1000, type=int,
                    help='The num of last fc layers for using softmax')


parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr-drop-epoch', default=[30, 60, 90], type=int, nargs='+',
                    help='The learning rate drop epoch')
parser.add_argument('--lr-drop-ratio', default=0.1, type=float,
                    help='The learning rate drop ratio')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--pth-save-fold', default='tmp', type=str,
                    help='The folder to save pths')
parser.add_argument('--pth-save-epoch', default=1, type=int,
                    help='The epoch to save pth')


# magface parameters
parser.add_argument('--l_a', default=10, type=float,
                    help='lower bound of feature norm')
parser.add_argument('--u_a', default=110, type=float,
                    help='upper bound of feature norm')
parser.add_argument('--l_margin', default=0.45,
                    type=float, help='low bound of margin')
parser.add_argument('--u_margin', default=0.8, type=float,
                    help='the margin slop for m')
parser.add_argument('--lambda_g', default=20, type=float,
                    help='the lambda for function g')
parser.add_argument('--arc-scale', default=64, type=int,
                    help='scale for arcmargin loss')
parser.add_argument('--vis_mag', default=1, type=int,
                    help='visualize the magnitude against cos')

args = parser.parse_args()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="face-rec-models",
    name="magface-fine-tuner",
    config={"lr": args.lr, "epochs": args.epochs}
)

    #track hyperparameters and run metadata
    #config={args})

def load_dict_finetuner(args, model):
    """
    Function for loading pre-trained model weights (from MagFace network_inf.py)
    Model weights, optimizer etc. are stored in checkpoint['state_dict']
    Pre-trained model in this thesis is: iresnet18, loaded from magface.builder
    Updates this model with the weights from a model from a checkpoint. 
    Returns model 
    """
    if os.path.isfile(args.pretrained): #pretrained should be checkpoint path!
        cprint('=> loading pth from {} ...'.format(args.pretrained), 'green')
        if args.cpu_mode:
            print("cpu!!!")
            checkpoint = torch.load(args.pretrained, map_location=torch.device("cpu")) #pretrained should be checkpoint path!
        else:
            checkpoint = torch.load(args.pretrained, map_location=torch.device('cuda'))
        
        _state_dict = clean_dict_finetuner(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        for name, param in _state_dict.items():
            model_dict[name].copy_(param.to('cuda'))  # Ensure parameters are on the correct device
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        
        #model_dict.update(_state_dict)
        #model.load_state_dict(model_dict)
        
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        print("sys exit")
        sys.exit("=> No checkpoint found at '{}'".format(args.pretrained)) #pretrained should be checkpoint path!
    return model


def clean_dict_finetuner(model, state_dict):
    """Function used in load_dict. This function filters out unnecessary 
    keys from the loaded state dictionary based on the model's state dictionary. """
    i = 0
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        # # assert k[0:1] == 'features.module.'
        new_k = 'features.'+'.'.join(k.split('.')[2:])
        i += 1

        if new_k in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        else:
            print("K? ", new_k)
            #print(v.size())
            #print()
            #print(k,i, "III, did not happen",v.size(), model.state_dict()[new_k].size(), "\nhvordan ser v ud V", v ) 
        # assert k[0:1] == 'module.features.'
        new_kk = '.'.join(k.split('.')[1:])
        if new_kk in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
        else:
            print("new KK 2", new_kk)
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        if num_model == num_ckpt + 1:
          print(f'=> fc.weight not loaded! Model params: {num_model}, loaded params: {num_ckpt}')
        else:  
          sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
            num_model, num_ckpt))
    return _state_dict

def unfreeze_last_layers_by_type(model, num_layers_to_unfreeze):
    # Identify layers to unfreeze (Conv2d and Linear layers)
    layers_to_unfreeze = []
    layer_types = (torch.nn.Conv2d, torch.nn.Linear)
    
    # List of layers to exclude from unfreezing
    exclude_layers = ['module.features.layer3.0.downsample.0', 'module.features.layer4.0.downsample.0']
    
    # Get all layers
    for name, module in model.named_modules():
        if isinstance(module, layer_types) and name not in exclude_layers:
            layers_to_unfreeze.append(name)
    
    # Determine the last num_layers_to_unfreeze layers
    layers_to_unfreeze = layers_to_unfreeze[-num_layers_to_unfreeze:]
    print(f"Layers to unfreeze: {layers_to_unfreeze}")
    
    # Unfreeze the selected layers
    for name, param in model.named_parameters():
        layer_name = name.rsplit('.', 1)[0]
        if layer_name in layers_to_unfreeze:
            param.requires_grad = True
            print(f"Unfroze layer: {layer_name}")



def main(args):
    "from trainer.py"
    # check the feasible of the lambda g - magface settings
    s = 64
    k = (args.u_margin-args.l_margin)/(args.u_a-args.l_a)
    min_lambda = s*k*args.u_a**2*args.l_a**2/(args.u_a**2-args.l_a**2)
    color_lambda = 'red' if args.lambda_g < min_lambda else 'green'
    cprint('min lambda g is {}, currrent lambda is {}'.format(
        min_lambda, args.lambda_g), color_lambda)

    cprint('=> torch version : {}'.format(torch.__version__), 'green')
    ngpus_per_node = torch.cuda.device_count()
    cprint('=> ngpus : {}'.format(ngpus_per_node), 'green')

    main_worker(args)


def main_worker(args):
    global best_acc1
    #print("Value of args.pretrained:", args.pretrained)  # Add this line for debugging

    if args.pretrained:
        
        cprint('=> modeling the network ...', 'green')
        model = magface.builder(args)
        #print("MODEL1:", model)
        #num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print("Number of trainable parameters: MODEL 1", num_params)
        #model = load_dict_inf(args, model) # loading using pre-trained model (as in network_inf.py)
        
        # prints which parameters are being loaded - not the fc layer..
        # returns model... i tvivl om der skal være = tegn.
        load_dict_finetuner(args, model)
        #print("MODEL2:", model)
        # Print number of parameters
        #num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print("Number of trainable parameters: MODEL 2", num_params)
        model = torch.nn.DataParallel(model).cuda() #.to('cpu')#.cuda()
        #model = torch.nn.DataParallel(model).to(device) # if no gpu

        ##### HERTIL
        
        #print("MODEL3:")
        # for name, param in model.named_parameters():
        #     cprint(' : layer name and parameter size - {} - {}'.format(name, param.size()), 'green')
        
        cprint('=> FREEZING ALL LAYERS EXCEPT fc ...', 'green')
        # Freezing all layers except the last layer
        for name, param in model.named_parameters():
            if not name.startswith('module.fc'):  # Stating the earlier layers does not need to get updated
                param.requires_grad = False
            else:
                print("NAME FC?", name)
        

        cprint('=> building the optimizer ...', 'green')
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        pprint.pprint(optimizer)

        cprint('=> building the dataloader ...', 'green')
        train_loader = dataloader.train_loader(args)
        print("TRAIN LOADER????", train_loader, "ARGS: are workers and batch size and train list"),

        cprint('=> building the criterion ...', 'green')
        criterion = magface.MagLoss(
            args.l_a, args.u_a, args.l_margin, args.u_margin)

        global iters
        iters = 0

        cprint('=> starting training engine ...', 'green')
        for epoch in range(args.start_epoch, args.epochs):

            global current_lr
            current_lr = utils.adjust_learning_rate(optimizer, epoch, args)
            
            # UNFREEZING LAST 9 LAYERS
            # In your training loop or wherever you handle the epoch logic
            if epoch == args.lr_drop_epoch[0]:
                # Unfreeze the last num_layers_to_unfreeze unique layers
                unfreeze_last_layers_by_type(model, num_layers_to_unfreeze=9)

                # Reinitialize the optimizer with the updated parameters
                optimizer = torch.optim.SGD(model.parameters(), current_lr, momentum=args.momentum, weight_decay=args.weight_decay)
                cprint('=> UNFREEZING SELECTED LAYERS...', 'green')

            # train for one epoch
            co2_emission, top1, top5, losses_id = do_train(train_loader, model, criterion, optimizer, epoch, args)
            #do_train(train_loader, model, criterion, optimizer, epoch, args)
                    
            print("LOSS ID:", losses_id)


            # save pth
            if epoch % args.pth_save_epoch == 0:
                state_dict = model.state_dict()

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                }, False,
                    filename=os.path.join(
                    args.pth_save_fold, '{}.pth'.format(
                        str(epoch+1).zfill(5))
                ))
                cprint(' : save pth for epoch {}'.format(epoch + 1))
                # log metrics to wandb
                wandb.log({"epochs": epoch ,"CO2 emission (in Kg)": co2_emission, "acc1": top1.avg, "acc5": top5.avg,"losses_id": losses_id.avg})
    else:
        print("args.pretrained is False")  # Add this line for debugging



def do_train(train_loader, model, criterion, optimizer, epoch, args):
    # create codecarbon tracker
    tracker = EmissionsTracker()
    tracker.start()

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.3f')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    learning_rate = utils.AverageMeter('LR', ':.4f')
    throughputs = utils.AverageMeter('ThroughPut', ':.2f')

    losses_id = utils.AverageMeter('L_ID', ':.3f')
    losses_mag = utils.AverageMeter('L_mag', ':.6f')
    progress_template = [batch_time, data_time, throughputs, 'images/s',
                         losses, losses_id, losses_mag, 
                         top1, top5, learning_rate]

    progress = utils.ProgressMeter(
        len(train_loader),
        progress_template,
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    print("LEN TRAIN LOADER:", len(train_loader))

    # update lr
    learning_rate.update(current_lr)
    model = model.to('cuda') #.to('cpu') #or cuda

    for i, (input, target) in enumerate(train_loader):
        print("---!!--I AND TARGET!!!!!-----!!-----", i, target)
        #print("--------TENSOR_SHAPE--------\nInput batch size:", input.size())
        #print("Target batch size:", target.max())

        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1
        
        
        input = input.cuda(non_blocking=True) #.to('cuda', non_blocking=True) #.cuda(non_blocking=True) #.to('cpu', non_blocking=True) #or cuda
        target = target.cuda(non_blocking=True)#.to('cuda', non_blocking=True) #.cuda(non_blocking=True) #.to('cpu', non_blocking=True) #or cuda

        #input = input.cuda(non_blocking=True)
        #target = target.cuda(non_blocking=True)
        
        #print("--------TENSOR_SHAPE--------\nInput batch size:Input batch size:", input.size())
        #print("Target batch size:", target.size())

        # compute output
        output, x_norm = model(input, target)
        
        ###
        # Debugging prints
        print(f"Output shape: {output[0].shape}")
        print(f"Target shape: {target.shape}, Target min: {target.min()}, Target max: {target.max()}, Output[0] size: {output[0].size(1)}")

        # Ensure all target indices are within the valid range
        #assert target.min() >= 0 and target.max() < output[0].size(1), "Target index out of bounds"

                
        
        #print(f"Output shape from model: {output.shape}")
        print(f"--------TENSOR_SHAPE--------\nx_norm shape from model: {x_norm.shape}")
        
        #output_squeezed = [torch.squeeze(tensor, dim=1) for tensor in output]
        print(f"--------TENSOR_SHAPE--------\noutput shape from model:")
        for idx, tensor in enumerate(output):
            print(f"Shape of tensor {idx}: {tensor.shape}")
              

        loss_id, loss_g, one_hot = criterion(output, target, x_norm)
        loss = loss_id + args.lambda_g * loss_g
        print("AFTER LOSS")

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(args, output[0], target, topk=(1, 5))
        print("here")

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        losses_id.update(loss_id.item(), input.size(0))
        losses_mag.update(args.lambda_g*loss_g.item(), input.size(0))

        # compute gradient and do solver step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        duration = time.time() - end
        batch_time.update(duration)
        end = time.time()
        throughputs.update(args.batch_size / duration)
        print("here2")


        if i % args.print_freq == 0:
            progress.display(i)
            debug_info(x_norm, args.l_a, args.u_a,
                           args.l_margin, args.u_margin)

        if args.vis_mag:
            if (i > 10000) and (i % 100 == 0):
                x_norm = x_norm.detach().cpu().numpy()
                cos_theta = torch.masked_select(
                    output[0], one_hot.bool()).detach().cpu().numpy()
                logit = torch.masked_select(
                    F.softmax(output[0]), one_hot.bool()).detach().cpu().numpy()
                np.savez('{}/vis/epoch_{}_iter{}'.format(args.pth_save_fold, epoch, i),
                         x_norm, logit, cos_theta)
    emissions = tracker.stop()
    return emissions, top1, top5, losses_id


def debug_info(x_norm, l_a, u_a, l_margin, u_margin):
    """
    visualize the magnitudes and magins during training.
    Note: modify the function if m(a) is not linear
    """
    mean_ = torch.mean(x_norm).detach().cpu().numpy()
    max_ = torch.max(x_norm).detach().cpu().numpy()
    min_ = torch.min(x_norm).detach().cpu().numpy()
    m_mean_ = (u_margin-l_margin)/(u_a-l_a)*(mean_-l_a) + l_margin
    m_max_ = (u_margin-l_margin)/(u_a-l_a)*(max_-l_a) + l_margin
    m_min_ = (u_margin-l_margin)/(u_a-l_a)*(min_-l_a) + l_margin
    print('  [debug info]: x_norm mean: {:.2f} min: {:.2f} max: {:.2f}'
          .format(mean_, min_, max_))
    print('  [debug info]: margin mean: {:.2f} min: {:.2f} max: {:.2f}'
          .format(m_mean_, m_min_, m_max_))


if __name__ == '__main__':

    pprint.pprint(vars(args))
    main(args)
