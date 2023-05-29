"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import yaml
from termcolor import colored
from utils.common_config import get_val_dataset, get_val_transformations, get_val_dataloader,\
                                get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
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
FLAGS.add_argument('--clusterk', default=20, help='number of k clusters for scan method')
FLAGS.add_argument('--path_to_img', default="./data/cifar_img_5109.jpeg", help='sample img')
args = FLAGS.parse_args()

def main():
    
    # Read config file
    print(colored('Read config file {} ...'.format(args.config_pretext), 'blue'))
    with open(args.config_pretext, 'r') as stream:
        config_simclr = yaml.safe_load(stream)
    config_simclr['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config_simclr)
    print(colored('Read config file {} ...'.format(args.config_scan), 'blue'))
    with open(args.config_scan, 'r') as stream:
        config_scan = yaml.safe_load(stream)
    config_scan['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config_scan)

    # Get dataset for fine-tuning 
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_val_transformations(config_simclr)
    dataset = get_val_dataset(config_simclr, transforms)
    dataloader = get_val_dataloader(config_simclr, dataset)
    print('Number of samples: {}'.format(len(dataset)))

    img_sample = {"image":[np.asarray(Image.open(args.path_to_img))], "target":["unknown"]}

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

    # Perform evaluation
    print(colored('Perform evaluation of the pretext task (setup={}).'.format(config_simclr['setup']), 'blue'))
    print('Create Memory Bank')
    retrain_mb = MemoryBank(len(dataset), config_simclr['model_kwargs']['features_dim'],
                            config_simclr['num_classes'], config_simclr['criterion_kwargs']['temperature'])
    single_img = MemoryBank(1, config_simclr['model_kwargs']['features_dim'], 
                            config_simclr['num_classes'], config_simclr['criterion_kwargs']['temperature'])
    retrain_mb.cuda()
    single_img.cuda()

    print('Fill Memory Bank')
    fill_memory_bank(dataloader, simclr, retrain_mb)

    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(dataloader, simclr, retrain_mb)
    print('Mine the nearest neighbors (Top-%d)' %(args.topk)) 
    indices, acc = retrain_mb.mine_nearest_neighbors(args.topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(args.topk, 100*acc))
    np.save(os.path.join(args.save_path, 'pretext', 'top%d_neighbors_train_path'%args.topk))   

    
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(dataloader, simclr, single_img)
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = single_img.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(p['topk_neighbors_val_path'], indices)   


    retrain_mb = MemoryBank(len(dataset), config_simclr['model_kwargs']['features_dim'],
                            config_scan['num_classes'], config_scan['temperature'])
    single_img = MemoryBank(1, config_scan['model_kwargs']['features_dim'], 
                            config_scan['num_classes'], config_scan['temperature'])
    retrain_mb.cuda()
    single_img.cuda()
        

        print('Mine the nearest neighbors')
        for topk in [1, 5, 20]: # Similar to Fig 2 in paper 
            _, acc = memory_bank.mine_nearest_neighbors(topk)
            print('Accuracy of top-{} nearest neighbors on validation set is {:.2f}'.format(topk, 100*acc))

    elif config['setup'] in ['scan', 'selflabel']:
        print(colored('Perform evaluation of the clustering model (setup={}).'.format(config['setup']), 'blue'))
        head = state_dict['head'] if config['setup'] == 'scan' else 0
        predictions, features = get_predictions(config, dataloader, model, return_features=True)
        print("Dataloader Sample Keys:",next(iter(dataloader)).keys(),"| Dataloader Image Shape:",next(iter(dataloader))["image"].size())
        print("Predictions keys:",predictions[0].keys(),"| Predictions Shape:",predictions[0]["predictions"].size())
        print("Features Shape:",features.size())
        clustering_stats = hungarian_evaluate(head, predictions, dataset.classes, 
                                                compute_confusion_matrix=True)
        print(clustering_stats)
        if args.visualize_prototypes:
            prototype_indices = get_prototypes(config, predictions[head], features, model)
            visualize_indices(prototype_indices, dataset, clustering_stats['hungarian_match'])
    else:
        raise NotImplementedError

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
