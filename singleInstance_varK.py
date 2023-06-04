"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import yaml
from termcolor import colored
from utils.common_config import get_train_dataset, get_train_transformations, get_train_dataloader,\
                                get_model, get_val_dataloader
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
from sampleDataSet import SampleDataSet
from get_sample_img import get_pic
from PIL import Image
import numpy as np
import os 


FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--config_pretext', help='Location of simclr config file')
FLAGS.add_argument('--config_scan', help='Location of scan config file')
FLAGS.add_argument('--simclr', help='Location where model is saved')
FLAGS.add_argument('--scan', help='Location where model is saved')
FLAGS.add_argument('--visualize_prototypes', action='store_true', 
                    help='Show the prototpye for each cluster')
FLAGS.add_argument('--save_path', default='./results', help='Location of save_paths')
FLAGS.add_argument('--topk', default=60, help='top k number for knn simclr method')
FLAGS.add_argument('--cluster_head', default=1, help='number of k clusters for scan method')
FLAGS.add_argument('--path_to_img', default="./data/cifar_img_5109.jpeg", 
                   help='path to the sample img')
FLAGS.add_argument('--re_calc', default=False, action='store_true', 
                   help='re-calc knn for train set')
args = FLAGS.parse_args()

topk_sample_file_path = os.path.join(os.path.dirname(__file__), 'results', 'cifar-20', 
                                     'pretext', 'topk-single_img-neighbors.npy')
topk_train_file_path = os.path.join(os.path.dirname(__file__), 'results', 'cifar-20', 
                                    'pretext', 'topk-train-neighbors.npy')

def main():

    if args.re_calc:
        ''' Re-Calculation Section '''
        # Read config file
        print(colored('Read config file {} ...'.format(args.config_pretext), 'blue'))
        with open(args.config_pretext, 'r') as stream:
            config_simclr = yaml.safe_load(stream)
        config_simclr['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
        print(config_simclr)

        # Get dataset for fine-tuning 
        print(colored('Get train dataset ...', 'blue'))
        transforms = get_train_transformations(config_simclr)
        dataset = get_train_dataset(config_simclr, transforms)
        dataloader = get_train_dataloader(config_simclr, dataset)
        print('Number of samples: {}'.format(len(dataset)))

        # Get model
        print(colored('Get models ...', 'blue'))
        print("SimCLR:")
        simclr = get_model(config_simclr)
        print(simclr)

        # Read model weights
        print(colored('Load model weights ...', 'blue'))
        state_dict_simclr = torch.load(args.simclr, map_location='cpu')

        # CUDA
        simclr.cuda()

        # Perform re-calc for train set KNN 
        print(colored('Perform KNN re-calc of the train set for SimCLR pass-through (setup={}).'.format(config_simclr['setup']), 'blue'))
        print('Create Memory Bank')
        recalc_mb = MemoryBank(len(dataset), config_simclr['model_kwargs']['features_dim'], 
                               config_simclr['num_classes'], config_simclr['criterion_kwargs']['temperature'])
        recalc_mb.cuda()

        # Mine the topk nearest neighbors for the query image.
        print(colored('Fill memory bank for mining the nearest neighbors ...', 'blue'))
        fill_memory_bank(dataloader, simclr, recalc_mb)
        print('Mine the nearest neighbors (Top-%d)' %(args.topk)) 
        indices, acc = recalc_mb.mine_nearest_neighbors(args.topk)
        print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(args.topk, 100*acc))
        np.save( topk_train_file_path, indices )   


    ''' Sample Section '''
    img_samples = np.array([get_pic()[2] for _ in range(128)])
    #np.array([np.asarray(Image.open(args.path_to_img))])

    # Read config file
    print(colored('Read config file {} ...'.format(args.config_pretext), 'blue'))
    with open(args.config_pretext, 'r') as stream:
        config_simclr = yaml.safe_load(stream)
    config_simclr['batch_size'] = min(len(img_samples),512) # To make sure we can evaluate on a single 1080ti
    print(config_simclr)
    print(colored('Read config file {} ...'.format(args.config_scan), 'blue'))
    with open(args.config_scan, 'r') as stream:
        config_scan = yaml.safe_load(stream)
    config_scan['batch_size'] = min(len(img_samples),512) # To make sure we can evaluate on a single 1080ti
    print(config_scan)

    print(colored('Get query dataset ...', 'blue'))
    img_dataset = SampleDataSet(img_samples)
    img_dataloader = get_val_dataloader(config_simclr,img_dataset)
    print('Number of samples: {}'.format(len(img_dataset)))

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
    state_dict_simclr = torch.load(args.simclr, map_location='cpu')
    state_dict_scan = torch.load(args.scan, map_location='cpu')

    simclr.load_state_dict(state_dict_simclr)
    scan.load_state_dict(state_dict_scan['model'])
        
    # CUDA
    simclr.cuda()
    scan.cuda()

    # Perform KNN calc for img samples  
    print(colored('Perform KNN calc for given img samples after SimCLR pass-through (setup={}).'.format(config_simclr['setup']), 'blue'))
    print('Create Memory Bank')
    single_img = MemoryBank(len(img_dataset), config_simclr['model_kwargs']['features_dim'], 
                            config_simclr['num_classes'], config_simclr['criterion_kwargs']['temperature'])
    single_img.cuda()

    # Mine the topk nearest neighbors for the query image.
    print(colored('Fill memory bank for mining the nearest neighbors ...', 'blue'))
    fill_memory_bank(img_dataloader, simclr, single_img)
    print('Mine the nearest neighbors (Top-%d)' %(args.topk))
    indices, acc = single_img.mine_nearest_neighbors(args.topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(args.topk, 100*acc))
    np.save( topk_sample_file_path, indices ) 

    # SCAN evaluation
    print(colored('Perform evaluation of the clustering model (setup={}).'.format(config_scan['setup']), 'blue'))
    head = state_dict_scan['head'] if config_scan['setup'] == 'scan' else 0
    predictions, features = get_predictions(config_scan, img_dataloader, scan, return_features=True, cluster_head=args.cluster_head)
    print("Dataloader Sample Keys:",next(iter(img_dataloader)).keys(),"| Dataloader Image Shape:",next(iter(img_dataloader))["image"].size())
    print("Predictions keys:",predictions[0].keys(),"| Predictions Shape:",predictions[0]["predictions"].size())
    print("Features Shape:",features.size())
    clustering_stats = hungarian_evaluate(head, predictions, img_dataset.classes, 
                                          compute_confusion_matrix=True)
    print(clustering_stats)
    if args.visualize_prototypes:
        prototype_indices = get_prototypes(config_scan, predictions[head], features, scan)
        visualize_indices(prototype_indices, img_dataset, clustering_stats['hungarian_match'])


@torch.no_grad()
def get_prototypes(config, predictions, features, model, topk=10):
    import torch.nn.functional as F

    # Get topk most certain indices and pred labels
    print('Get topk')
    probs = predictions['probabilities']
    n_classes = probs.shape[1]
    dims = features.shape[1]
    max_probs, pred_labels = torch.max(probs, dim = 1)
    indices = torch.zeros((n_classes, topk))
    for pred_id in range(n_classes):
        probs_copy = max_probs.clone()
        mask_out = ~(pred_labels == pred_id)
        probs_copy[mask_out] = -1
        conf_vals, conf_idx = torch.topk(probs_copy, k = topk, largest = True, sorted = True)
        indices[pred_id, :] = conf_idx

    # Get corresponding features
    selected_features = torch.index_select(features, dim=0, index=indices.view(-1).long())
    selected_features = selected_features.unsqueeze(1).view(n_classes, -1, dims)

    # Get mean feature per class
    mean_features = torch.mean(selected_features, dim=1)

    # Get min distance wrt to mean
    diff_features = selected_features - mean_features.unsqueeze(1)
    diff_norm = torch.norm(diff_features, 2, dim=2)

    # Get final indices
    _, best_indices = torch.min(diff_norm, dim=1)
    one_hot = F.one_hot(best_indices.long(), indices.size(1)).byte()
    proto_indices = torch.masked_select(indices.view(-1), one_hot.view(-1))
    proto_indices = proto_indices.int().tolist()
    return proto_indices

def visualize_indices(indices, dataset, hungarian_match):
    import matplotlib.pyplot as plt
    import numpy as np

    for idx in indices:
        img = np.array(dataset.get_image(idx)).astype(np.uint8)
        img = Image.fromarray(img)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    main() 
