"""
Author: Nils Treuheit
"""
import argparse
import torch
import yaml
from termcolor import colored
from utils.common_config import get_train_dataset, get_train_transformations, get_train_dataloader,\
                                get_model, get_val_dataloader, get_val_transformations, get_criterion,\
                                get_optimizer, adjust_learning_rate, get_val_dataset
from utils.evaluate_utils import get_predictions, hungarian_evaluate, get_sample_preds
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
from utils.train_utils import simclr_train
from utils.evaluate_utils import contrastive_evaluate
from data.custom_dataset import NeighborsDataset
from sampleDataSet import SampleDataSet
from get_sample_img import get_pic
from PIL import Image
import numpy as np
import os 


FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--config_pretext', default="./configs/pretext/simclr_cifar20.yml", 
                   help='Location of simclr config file')
FLAGS.add_argument('--config_main', default="./configs/scan/scan_cifar20.yml", 
                   help='Location of scan config file')
FLAGS.add_argument('--simclr_model', default="./results/cifar-20/pretext/model.pth.tar",
                    help='Location where model is saved')
FLAGS.add_argument('--simclr_checkpoint', default="./results/cifar-20/pretext/checkpoint.pth.tar",
                    help='Location where model is saved')
FLAGS.add_argument('--scan_model', default="./results/cifar-20/scan/model.pth.tar",
                   help='Location where model is saved')
FLAGS.add_argument('--scan_checkpoint', default="./results/cifar-20/scan/checkpoint.pth.tar",
                   help='Location where model is saved')
FLAGS.add_argument('--save_path', default='./results', help='Location of save_paths')
FLAGS.add_argument('-k','--topk', default=50, help='top k number for knn simclr method')
FLAGS.add_argument('-c','--cluster_heads', default="0,2,4", 
                   help='comma separated index list of cluster heads'+
                   ' (each head defines a number of k clusters) '+
                   'for scan method (pretrained heads for k=[5,20,100,300,500])')
FLAGS.add_argument('-q','--paths_to_img', default="./data/cifar_img_5109.jpeg", 
                   help='paths to the sample img separated by comma no spaces')
FLAGS.add_argument('-e','--epochs', default=25, 
                   help='number of epochs for re-training phase')
FLAGS.add_argument('--no_viz', default=False, action='store_true', 
                   help='disable plot evaluation functionality')
FLAGS.add_argument('--perf', default=False, action='store_true', 
                   help='used to time minimal query effort method')
args = FLAGS.parse_args()

topk_sample_file_path = os.path.join(os.path.dirname(__file__), 'results', 'cifar-20', 
                                     'pretext', 'topk-single_img-neighbors.npy')
topk_train_file_path = os.path.join(os.path.dirname(__file__), 'results', 'cifar-20', 
                                    'pretext', 'topk-train-neighbors.npy')
pretext_checkpoint_path = os.path.join(os.path.dirname(__file__),"results","cifar-20",
                                       "pretext","checkpoint_%d.pth.tar"%args.topk)
pretext_model_path = os.path.join(os.path.dirname(__file__),"results","cifar-20",
                                  "pretext","model_%d.pth.tar"%args.topk)


def main():

    ''' Re-Train Section '''
    
    """ => SimCLR Re-Train """

    # Read config file
    print(colored('Read config file {} ...'.format(args.config_pretext), 'blue'))
    with open(args.config_pretext, 'r') as stream:
        config_simclr = yaml.safe_load(stream)
    config_simclr['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config_simclr)

    # Get dataset for fine-tuning 
    print(colored('Get train dataset ...', 'blue'))
    transforms = get_train_transformations(config_simclr)
    dataset = get_train_dataset(config_simclr, transforms, to_augmented_dataset=True,
                                split='train+unlabeled')
    dataloader = get_train_dataloader(config_simclr, dataset)
    print('Number of samples: {}'.format(len(dataset)))

    print(colored('Get validation dataset ...', 'blue'))
    val_transforms = get_val_transformations(config_simclr)
    val_dataset = get_val_dataset(config_simclr, val_transforms)
    val_dataloader = get_val_dataloader(config_simclr, val_dataset)
    print('Number of samples: {}'.format(len(val_dataset)))
    ## Dataset w/o augs for knn eval
    base_dataset = get_train_dataset(config_simclr, val_transforms, split='train') 
    base_dataloader = get_val_dataloader(config_simclr, base_dataset) 

    # Get model
    print(colored('Get models ...', 'blue'))
    print("SimCLR:")
    simclr = get_model(config_simclr)
    print(simclr)

    # Read model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict_simclr = torch.load(args.simclr_model, map_location='cpu')

    # CUDA
    simclr.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Perform re-calc for train set KNN 
    print(colored('Perform KNN re-calc of the train set for SimCLR pass-through (setup={}).'.format(config_simclr['setup']), 'blue'))
    print('Create Memory Bank')
    recalc_mb = MemoryBank(len(base_dataset), config_simclr['model_kwargs']['features_dim'], 
                           config_simclr['num_classes'], config_simclr['criterion_kwargs']['temperature'])
    recalc_mb.cuda()

    val_mb = MemoryBank(len(val_dataset),config_simclr['model_kwargs']['features_dim'],
                        config_simclr['num_classes'], config_simclr['criterion_kwargs']['temperature'])
    val_mb.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(config_simclr)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(config_simclr, simclr)
    print(optimizer)

    # re-train from checkpoint
    print(colored('Restart from checkpoint {}'.format(args.simclr_checkpoint), 'blue'))
    checkpoint = torch.load(args.simclr_checkpoint, map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    simclr.load_state_dict(checkpoint['model'])
    simclr.cuda()
    start_epoch = checkpoint['epoch']

    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, args.epochs):
        print(colored('Epoch %d/%d' %(epoch, args.epochs), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(config_simclr, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')
        simclr_train(dataloader, simclr, criterion, optimizer, epoch)

        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, simclr, recalc_mb)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, simclr, recalc_mb)
        print('Result of kNN evaluation is %.2f' %(top1)) 
        
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': simclr.state_dict(), 
                    'epoch': epoch + 1}, pretext_checkpoint_path)

    # Save final model
    torch.save(simclr.state_dict(), pretext_model_path)

    # Mine the topk nearest neighbors for the query image.
    print(colored('Fill memory bank for mining the nearest neighbors ...', 'blue'))
    fill_memory_bank(dataloader, simclr, recalc_mb)
    print('Mine the nearest neighbors (Top-%d)' %(args.topk)) 
    indices, acc = recalc_mb.mine_nearest_neighbors(args.topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(args.topk, 100*acc))
    np.save( topk_train_file_path, indices )   

    """ => SCAN Re-Train """

    

    ''' Sample Section '''
    # Read config file
    print(colored('Read config file {} ...'.format(args.config_pretext), 'blue'))
    with open(args.config_pretext, 'r') as stream:
        config_simclr = yaml.safe_load(stream)
    config_simclr['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config_simclr)
    print(colored('Read config file {} ...'.format(args.config_main), 'blue'))
    with open(args.config_main, 'r') as stream:
        config_scan = yaml.safe_load(stream)
    config_scan['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config_scan)

    # Get model
    print(colored('Get models ...', 'blue'))
    print("SimCLR:")
    simclr = get_model(config_simclr)
    print(simclr)
    print("SCAN:")
    scan = get_model(config_scan)
    print(scan)

    # Read model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict_simclr = torch.load(args.simclr_model, map_location='cpu')
    state_dict_scan = torch.load(args.scan_model, map_location='cpu')

    simclr.load_state_dict(state_dict_simclr)
    scan.load_state_dict(state_dict_scan['model'])
    
    # CUDA
    simclr.cuda()
    scan.cuda()

    print(colored('Get query dataset ...', 'blue'))
    img_transform = get_val_transformations(config_simclr)
    img_samples = np.array([np.asarray(Image.open(img_path)) for img_path in args.paths_to_img.split(',')])
    img_dataset = SampleDataSet(img_samples,transform=img_transform)
    img_dataloader = get_val_dataloader(config_simclr,img_dataset)
    print('Number of samples: {}'.format(len(img_dataset)))

    # Perform KNN calc for img samples  
    print(colored('Perform KNN calc for given img samples after SimCLR pass-through (setup={}).'.format(config_simclr['setup']), 'blue'))
    print('Create Memory Bank')
    img_db = MemoryBank(len(img_dataset), config_simclr['model_kwargs']['features_dim'], 
                            config_simclr['num_classes'], config_simclr['criterion_kwargs']['temperature'])
    img_db.cuda()

    # Mine the topk nearest neighbors for the query image.
    print(colored('Fill memory bank for mining the nearest neighbors ...', 'blue'))
    fill_memory_bank(img_dataloader, simclr, img_db)
    print('Mine the nearest neighbors (Top-%d)' %(args.topk))
    indices, acc = img_db.mine_nearest_neighbors(args.topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(args.topk, 100*acc))
    if not args.perf: np.save( topk_sample_file_path, indices ) 

    img_transform = get_val_transformations(config_scan)
    img_dataset = NeighborsDataset(SampleDataSet(img_samples,transform=img_transform), indices, args.topk)
    img_dataloader = get_val_dataloader(config_scan,img_dataset)

    # SCAN evaluation
    cluster_heads = [int(cluster_head) for cluster_head in args.cluster_heads.split(",")]
    print(colored('Perform evaluation of the clustering model (setup={}).'.format(config_scan['setup']), 'blue'))
    if not args.perf:
        # get all heads predictions - to save time cluster head can also be given straight to get_prediction function
        predictions, features = get_predictions(config_scan, img_dataloader, scan, return_features=True)
        print("Features Shape:",features.size())
        # give results for specific heads
        for cluster_head in cluster_heads:
            print("%d. Cluster Head\n----------------"%cluster_head)
            print("Predictions keys:",predictions[cluster_head].keys(),
                  "| Predictions Shape:",predictions[cluster_head]["predictions"].size())
        
    # give eva for 20 cluster head since it is interpretable
    if not (args.no_viz or args.perf):
        clustering_stats = hungarian_evaluate(1, predictions, img_dataset.dataset.classes[:-1], 
                                              compute_confusion_matrix=True)
        print(clustering_stats)

    # give results for specific heads
    predictions, features = get_sample_preds(config_scan, img_dataloader, scan, return_features=True)
    print("Features of Sample/s:",features)
    for cluster_head in cluster_heads:
        print("Predictions of Sample/s for "+
              "%d. cluster head:"%cluster_head,
              predictions[cluster_head])


if __name__ == "__main__":
    main() 
