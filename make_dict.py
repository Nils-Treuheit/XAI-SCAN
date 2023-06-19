from utils.evaluate_utils import get_predictions, hungarian_evaluate, get_sample_preds
from data.custom_dataset import NeighborsDataset
from utils.memory import MemoryBank 
from utils.common_config import get_model
import yaml
import pickle
from utils.common_config import get_val_transformations, get_val_dataloader, get_train_dataloader,\
get_train_transformations, get_train_dataset,    get_visualization_transformations
import numpy as np
from sampleDataSet import SampleDataSet
from PIL import Image
import torch
from utils.cluster_visualization import (
    get_nearest_neighbours_for_image,
    get_clustering_stats,
    visualize_cluster,
    save_predictions,
    get_prototypes,
    find_most_common_words,
    show_wordcloud,
    get_caption,
)
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

with open("./configs/pretext/simclr_cifar20.yml", 'r') as stream:
    config_simclr = yaml.safe_load(stream)
    
with open("./configs/scan/scan_cifar20.yml", 'r') as stream:
    config_scan = yaml.safe_load(stream)
    
config_scan['batch_size'] = 512
config_scan['topk_neighbors_train_path'] = "results/cifar-20/pretext/topk-train-neighbors.npy"


transforms = get_visualization_transformations(config_scan)
dataset = get_train_dataset(config_scan, transforms, to_neighbors_dataset=False)
dataloader = get_train_dataloader(config_scan, dataset)


scan = get_model(config_scan)
scan.to("mps")
model_path = "/Users/mariomark/Desktop/XAI-SCAN/results/cifar-20/scan/model.pth.tar"
state_dict = torch.load(model_path, map_location="cpu")
scan.load_state_dict(state_dict["model"])

predictions_list, features = torch.load("predictions.pt")


# loops over the head

results = {}
image_captions = {}

def find_matching_indexes(array, number):
    return [index for index, value in enumerate(array) if value == number]

for i, predictions_dict in enumerate(predictions_list):
    # Get prototypes
    prototype_indices = get_prototypes(config_scan, predictions_dict, features, scan)
    
    # Get number of clusters
    num_clusters  = len(prototype_indices)
    
    # Make head key 
    head_name = f"head-{num_clusters}"
    
    results[head_name] = {}
    
    # Getting the predictions for this head
    predictions_for_head = predictions_dict["predictions"]
    predictions_for_head_neighbours = predictions_dict["neighbors"]    
    
    # Loop over clusters 
    for cluster in range(num_clusters):
        
        # Set cluster key 
        cluster_name = f"cluster-{cluster}"
        
        # Find indexes of images for this cluster
        indexes_for_cluster = find_matching_indexes(predictions_for_head, cluster)
        
        images = [Image.fromarray(dataset.get_image(i)) for i in indexes_for_cluster]
        
        print(f"Cluster-{cluster}", len(indexes_for_cluster))
        # Find 20 top 
        prototype = prototype_indices[cluster]
        prototype_neighbours = predictions_for_head_neighbours[prototype]
        
        # Finds captions for top k-s in cluster
        captions = [
            get_caption(image=Image.fromarray(dataset.get_image(i))) 
            if i not in image_captions
            else image_captions[i.item()]
            for i in prototype_neighbours
        ]
        for index,caption in zip(prototype_neighbours,captions):
            image_captions[index.item()] = caption
        
        # Builds a key value pair with words and number of times met in captions
        most_common_words = find_most_common_words(captions)
        print(most_common_words)
        # print("Center: ",prototype )
        # print("Neighbours: ",predictions_for_head)
        explanation = ''
        for word, count in most_common_words[:3]:
            explanation = explanation + ' ' + word 
        print(explanation)
        
        results[head_name][cluster_name] = {
            "images": images, 
            "top_neighbours": prototype_neighbours,
            "explanation": ""
        }
        with open('clusters_explanations.pkl', 'wb') as fp:
            pickle.dump(results, fp)
            print('dictionary saved successfully to file')
    