# Inbuilt 
import argparse
import csv
import json
import os
import pdb
import pprint
import time
from datetime import datetime

import numpy as np
import pycls.utils.logging as lu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# local
import models
import pycls.core.builders as model_builder
import pycls.core.optimizer as optim
from pycls.core.config import cfg
from pycls.datasets.data import Data
import pycls.utils.metrics as mu

from active_sampling.query_methods import active_sample

from train_test import test, train, train_mixup

logger = lu.get_logger(__name__)

plot_epoch_xvalues = []
plot_epoch_yvalues = []
plot_it_x_values = []
plot_it_y_values = []

def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)

    return parser


def main(cfg):

    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Getting the output directory ready (default is "/output")
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME)
    if not os.path.exists(dataset_out_dir):
        os.mkdir(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}+{now.month}+{now.day}+{now.hour}+{now.minute}+{now.second}'
    else:
        exp_dir = cfg.EXP_NAME

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    else:
        logger.info("=============================")
        logger.info("Experiment directory already exists: {}. Reusing it may lead to loss of old logs in the directory.".format(exp_dir))
        logger.info("=============================")

    # Dataset preparing steps
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)

    logger.info("=============================")
    logger.info("Dataset loaded sucessfully. Train Size: {} and Test Size: {}".format(train_size, test_size))
    logger.info("=============================")
    
    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INITAL_L_RATIO, \
        val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RND_SEED, save_dir=exp_dir)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)

    # Preparing dataloaders for initial training
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    uSet_loader = data_obj.getIndexesDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RND_SEED)

    # Initialize the model.  
    model = model_builder.build_model(cfg)
    logger.info("========MODEL========")
    logger.info("model: {}".format(cfg.MODEL.TYPE))
    logger.info("=========================")

    # Construct the optimizer
    optimizer = construct_optimizer(cfg, model)

    logger.info("========OPTIMIZER========")
    logger.info("optimizer: {}".format(optimizer))
    logger.info("=========================")

    # START AL
    #   - Train
    #   - Active Sample
    #   - Repeat




    for i in range(0, cfg.ACTIVE_LEARNING.MAX_ITER):

        # Creating output directory for the episode
        episode_dir = os.path.join(exp_dir, f'episode_{i}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)

        # Train model
        start_epoch = 1

        for epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH + 1):
            train_epoch()
        # Save best model in the episode directory

        # Active Sample 

        # Save lSet, uSet and activeSet in the episode directory




    # Log everything
    
    # train and test
    temp_best_val_acc = 0
    temp_best_val_epoch = 0
    best_model_state = None
    best_opt_state = None

    for cur_epoch in range(1, cfg.OPTIM.MAX_EPOCH + 1):
        model = train(cfg, model, device, train_loader, optimizer, cur_epoch, writer)
        # val_acc = test(cfg, model, device, val_loader, writer, 0, is_val=True)

        if temp_best_val_acc < val_acc:
            temp_best_val_acc = val_acc
            temp_best_val_epoch = cur_epoch

            #Save best model and optimizer state for checkpointing
            model.eval()

            best_model_state = model.state_dict()
            best_opt_state = optimizer.state_dict()
            model.train()

    checkpoint = {
    'epoch': temp_best_val_epoch,
    'model_state': best_model_state,
    'optimizer_state': best_opt_state}

    # Write the checkpoint
    # checkpoint_file = os.path.join(dest_dir_name, 'checkpoint_{}.pyth'.format(temp_best_val_epoch))
    # torch.save(checkpoint, checkpoint_file)

    model.load_state_dict(best_model_state)
    accuracy = test(args, model, device, test_loader, writer, 0)   
    
    # save model
    # model.eval()
    # save_path = os.path.join(dest_dir_name,'init.pth')
    # torch.save(model.state_dict(), save_path)
    # print("initial pool model saved in: ",save_path)

    # Save config.yaml as a json file inside the experiment directory
    with open(f'{exp_dir}/config.json', 'w') as filehandler:
        json.dump(cfg, filehandler)
    
    # save logs
    helper.utils.log(dest_dir_name, 0, args.sampling_method, 0, accuracy, [0]*args.init_size)
    helper.utils.log_picked_samples(dest_dir_name, np.genfromtxt(labeled_csv, delimiter=',', dtype=str))

    # start the active learning loop.
    episode_id = 1
    while True:
        if episode_id > args.max_eps:
            break
        # SAVE LABELED AND UNLABELED BEFORE STARTING SIMULATION
        # utils.save_lbl_unlbl(dest_dir_name)

        # read the unlabeled file
        unlabeled_rows = np.genfromtxt(unlabeled_csv, delimiter=',', dtype=str)
        labeled_rows = np.genfromtxt(labeled_csv, delimiter=',', dtype=str)
        print("Episode #", episode_id)
        # print("Epsilon ", epsilon)

        # sanity checks
        if len(unlabeled_rows) == 0:
            break

        # set the sample size
        sample_size = args.al_batch_size
        if len(unlabeled_rows) < sample_size:
            sample_size = len(unlabeled_rows)

        # sample
        sample_start = time.time()
        sample_rows = active_sample(args, unlabeled_rows, sample_size, data_transforms, method=args.sampling_method, model=model, dest_dir=dest_dir_name)
        sample_end = time.time()

        # log picked samples
        helper.utils.log_picked_samples(dest_dir_name, sample_rows, episode_id)

        sample_time = sample_end - sample_start

        # update the labeled pool
        labeled_rows = np.concatenate((labeled_rows,sample_rows),axis=0)
        np.savetxt(labeled_csv, labeled_rows,'%s',delimiter=',')

        # update the unlabeled pool
        unlabeled_rows = helper.utils.remove_rows(unlabeled_rows, sample_rows)
        np.savetxt(unlabeled_csv, unlabeled_rows, '%s', delimiter=',')

        print("Unlabeled pool size: ",len(unlabeled_rows))
        print("Labeled pool size: ",len(labeled_rows))


        #train the model
        dataset_train, sampler = data_obj.get_filtered_dataset('labeled', dest_dir=dest_dir_name)
        train_loader = DataLoader(dataset_train, sampler=sampler, batch_size=args.batch_size, shuffle=False, **kwargs)

        # initialize the model.  
        if args.model == 'lenet':
            model = Net().to(device)
        elif args.model == 'vgg16':
            model = vgg16(num_classes=args.nclasses).to(device)
        elif args.model == 'resnet18':
            model = resnet18(num_classes=args.nclasses).to(device)

        
        optimizer = optim.Adam(model.parameters(),lr=args.lr)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) # setup the optimizer
        temp_best_val_acc = 0
        temp_best_val_epoch = 0
        best_model_state = None
        best_opt_state = None
        
        for epoch in range(1, args.epochs + 1):
            model = train(args, model, device, train_loader, optimizer, epoch, writer)
                
            val_acc = test(args, model, device, val_loader, writer, episode_id, is_val=True)
            
            if temp_best_val_acc < val_acc:
                temp_best_val_acc = val_acc
                temp_best_val_epoch = epoch

                #Save best model and optimizer state for checkpointing
                model.eval()
                
                best_model_state = model.state_dict()
                best_opt_state = optimizer.state_dict()
                model.train()
        
        checkpoint = {
        'epoch': temp_best_val_epoch,
        'model_state': best_model_state,
        'optimizer_state': best_opt_state}
        
        # Write the checkpoint
        checkpoint_file = os.path.join(dest_dir_name, 'checkpoint_{}.pyth'.format(temp_best_val_epoch))
        torch.save(checkpoint, checkpoint_file)
        
        model.load_state_dict(best_model_state)
        accuracy = test(args, model, device, test_loader, writer, episode_id)   
        # save model
        # save_path = os.path.join(dest_dir_name, 'ep_'+str(episode_id)+'_'+str(epsilon)+'.pth')
        # torch.save(model.state_dict(), save_path)
        
        helper.utils.log(dest_dir_name, episode_id, args.sampling_method, sample_time, accuracy, labeled_rows, epsilon=temp_best_val_epoch)

        # # RESET LABELED AND UNLABELED TO HOW THEY WERE BEFORE SIMULATION
        # if epsilon != 0:
        #     helper.utils.rollback_lbl_unlbl(dest_dir_name)
        episode_id += 1
        temp_best_val_acc = 0
        temp_best_val_epoch = 0
        best_model_state = None
        best_opt_state = None



def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""

    # Shuffle the data
    #loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    len_train_loader = len(train_loader)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs
        # if cfg.NUM_GPUS > 1:
        #     #Average error and losses across GPUs
        #     #Also this this calls wait method on reductions so we are ensured
        #     #to obtain synchronized results
        #     loss, top1_err = du.scaled_all_reduce(
        #         [loss, top1_err]
        #     )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        # #Only master process writes the logs which are used for plotting
        # if du.is_master_proc():
        if True:
            if cur_iter is not 0 and cur_iter%20 == 0:
                #because cur_epoch starts with 0
                plot_it_x_values.append((cur_epoch)*len_train_loader + cur_iter)
                plot_it_y_values.append(loss)
                save_plot_values([plot_it_xvalues, plot_it_y_values],["plot_it_xvalues.npy", "plot_it_y_values.npy"], isDebug=False)
                #Plot loss graphs
                plot_arrays(x_vals=plot_it_xvalues, y_vals=plot_it_y_values, x_name="Iterations", y_name="Loss", dataset_name=cfg.TRAIN.DATASET)

        #Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return loss, clf_iter_count

@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.
    
    for cur_iter, (inputs, labels) in enumerate(test_loader):    
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        inputs = inputs.type(torch.cuda.FloatTensor)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err = du.scaled_all_reduce([top1_err])
            #as above returns a list
            top1_err = top1_err[0]
        # Copy the errors from GPU to CPU (sync point)
        top1_err = top1_err.item()
        # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
        misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
        totalSamples += inputs.size(0)*cfg.NUM_GPUS
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
        )
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications/totalSamples


if __name__ == "__main__":
    cfg.merge_from_file(argparser().parse_args().cfg_file)
    main(cfg)
