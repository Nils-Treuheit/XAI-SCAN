import torch
import yaml
from collections import defaultdict
from data.custom_dataset import NeighborsDataset
from sampleDataSet import SampleDataSet
from utils.common_config import (
    get_val_dataset,
    get_val_transformations,
    get_val_dataloader,
    get_model,
    get_train_dataset,
    get_visualization_transformations,
)
from utils.evaluate_utils import get_predictions, hungarian_evaluate
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

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Can be between 1 and topk (20)
#center_image_index = 15
config_path = "configs/scan/scan_cifar20.yml"
model_path = "results/cifar-20/scan/model.pth.tar"

config_key = "topk_neighbors_train_path"
top_k_file_path = "results/cifar-20/pretext/topk-train-neighbors.npy"


device = torch.device("cuda")


# The images shown are from the cluster of the center_image_index
show_images_for_single_cluster = False #True

# Show the images from multiple clusters randomly or from prototypes
show_images_multiple_clusters = True
show_from_prototypes = True #False

# Which prototype clusters to show in effect if show_from_prototypes== True
# Have to be 5
prototype_indices_to_show = [11, 12, 13, 14, 15]

# Make predictions for the dataset in order to run the clustering
# Done once the variable can be set to false to save time
making_predictions = True #False

# Finds captions for the neigbours of the center image
# Change to true if you change the center_image_index
find_captions = True #False

# If yoy change from_validation you have to set make_predictions to true
from_validation = False


def main():
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
        
    # Set the top_k path in the configuration with key and path 
    config[config_key] = top_k_file_path
    
    transforms = get_visualization_transformations(config)

    dataset = get_val_dataset(config, transforms, to_neighbors_dataset=True) if from_validation else \
            get_train_dataset(config, transforms, to_neighbors_dataset=True)

    # Only needed if we run the predictions again and not from file
    if making_predictions:
        dataloader = get_val_dataloader(config, dataset)

    model = get_model(config)
    model.to(device)
    state_dict = torch.load(model_path, map_location="cuda")
    model.load_state_dict(state_dict["model"])
    head = state_dict["head"]
    print("Model Heads count:", head)
    
    
    if making_predictions:
        predictions_list, features = save_predictions(config, dataloader, model)
        torch.save((predictions_list, features), "predictions.pt")
    else:
        predictions_list, features = torch.load("predictions.pt")

    print("Length of predictions: ", len(predictions_list))
    # print("Dataloader Sample Keys:",next(iter(dataloader)).keys())#,"| Dataloader Image Shape:",next(iter(dataloader))["image"].size())
    # print("Predictions keys:",predictions[0].keys(),"| Predictions Shape:",predictions[0]["predictions"].size())
    # print("Neighbours: ", predictions[0]["neighbors"].shape)
    # print("Predictions (clusters): ", predictions[0]["predictions"][1:10])

    # get_clustering_stats(predictions, head)
    print("Preds List: " ,predictions_list[0].keys())
    # Show the distribution of the clusters
    for predictions in predictions_list:
        
        clusters = defaultdict(list)
        
        for idx, c in enumerate(predictions["predictions"]):
            clusters[c.item()].append(idx)

        prototype_indices = get_prototypes(config, predictions, features, model)

        print("Top classified prototypes: ", prototype_indices)


        images_for_cluster_idx = get_nearest_neighbours_for_image(
            0, predictions
        )

        print("Number of neighgbours found: ", len(images_for_cluster_idx))

        # target cluster
        cluster = predictions["predictions"][0]
        print("Cluster of center: ", cluster.item())

        if find_captions:
            captions = visualize_cluster(
                images_for_cluster_idx, dataset, cluster, show_images_for_single_cluster
            )
            torch.save(captions, "captions.pt")
        else:
            captions = torch.load("captions.pt")

        # display Images
        if show_images_multiple_clusters:
            nrows = 5
            ncols = 6

            plt.figure(figsize=(8, 8))
            position = 1
            for i in prototype_indices_to_show:
                # Show random clusters or from prototyoes based on show_from_prototypes
                anchor_idx = (
                    prototype_indices[i]
                    if show_from_prototypes
                    else np.random.choice(range(predictions["predictions"].shape[0]))
                )

                neighbour_indicies = predictions["neighbors"][anchor_idx][1:6]
                cluster = predictions["predictions"][anchor_idx]
                print("anchor_idx: ", anchor_idx)
                print("Cluster: ", cluster.item())
                indices = [anchor_idx] + neighbour_indicies.tolist()

                for j in range(ncols):
                    plt.subplot(nrows, ncols, position)
                    plt.imshow(dataset.get_image(indices[j])["anchor"])
                    caption = get_caption(image=dataset.get_image(indices[j])["anchor"])
                    plt.title(caption, fontsize=8)
                    plt.axis("off")
                    position += 1
            print("Finished plot!")
            plt.show()

        most_common_words = find_most_common_words(captions)
        show_wordcloud(captions)
        for c in set(clusters.keys()):
            print("Cluster", c, ":", len(clusters[c]))
            
        for word, count in most_common_words[:10]:
            print(word, count)



def query_text_explain(predictions_list, features, dataset, predictions_complete, show_mc = True, show_wc = True, show_rand=True, use_pre_save=True):
    

    print("Preds List: " ,predictions_list[0].keys())
    # Show the distribution of the clusters
    for ch, preds_tuple in enumerate(zip(predictions_list,predictions_complete)):
        
        predictions, complete_preds = preds_tuple
        
        print(type(predictions["predictions"]),len(predictions["predictions"]))
        print("Visualize Query Images:")
        for idx,_ in enumerate(predictions["predictions"]):
            img = dataset.dataset.get_sample_image(idx) if(isinstance(dataset,NeighborsDataset)) else dataset.get_sample_image(idx)
            plt.imshow(img)
            caption = get_caption(image=Image.fromarray(img))
            plt.title(caption, fontsize=8)
            plt.axis("off")
            plt.title("Query Image",idx)
            plt.show()


        clusters = defaultdict(list)
        
        for idx, c in enumerate(predictions["predictions"]):
            clusters[c.item()].append(idx)

        prototype_indices = get_prototypes(complete_preds, features)

        print("Top classified prototypes:", prototype_indices)


        # Query Clusters
        captions = list()
        for idx,cluster in enumerate(predictions["predictions"]):
            images_for_cluster_idx = get_nearest_neighbours_for_image(
                idx, predictions, complete_preds
            )
            print("Number of neighbors found:", len(images_for_cluster_idx))
            
            print("KNN-Cluster of Query Image:", cluster.item())
            if use_pre_save and os.path.exists(f"captions_head{ch}_q{idx}.pt"):
                captions.append(torch.load(f"captions_head{ch}_q{idx}.pt"))
            else:    
                caps = visualize_cluster(
                    images_for_cluster_idx, dataset, cluster
                )
                torch.save(caps, f"captions_head{ch}_q{idx}.pt")
                captions.append(caps)
            

        # display Images
        if show_mc:
            indexes = np.random.choice([*range(len(prototype_indices))],5,replace=False)
            nrows = len(prototype_indices) if not show_rand else 5
            ncols = 6

            print("Show clusters:", indexes.tolist())

            plt.figure(figsize=(8, 8))
            position = 1
            proto_caps = list()
            for i in range(len(prototype_indices)):
                # Show random clusters or from prototyoes based on show_from_prototypes
                anchor_idx = prototype_indices[i]

                neighbour_indicies = complete_preds["neighbors"][anchor_idx][1:6]
                cluster = complete_preds["predictions"][anchor_idx]
                print("anchor_idx:", anchor_idx)
                print("Cluster:", cluster.item())
                indices = [anchor_idx] + neighbour_indicies.tolist()

                caps = list()
                for j in range(ncols):
                    img = dataset.dataset.get_image(indices[j]) if(isinstance(dataset,NeighborsDataset)) else dataset.get_image(indices[j])
                    caption = get_caption(image=Image.fromarray(img))
                    caps.append(caption)
                    if i in indexes:
                        plt.subplot(nrows, ncols, position)
                        plt.imshow(img)
                        plt.title(caption, fontsize=8)
                        plt.axis("off")
                        position += 1
                proto_caps.append(caps)
            print("Finished plot!")
            plt.title(f"{nrows} Random Clusters of Cluster Head {ch}")
            plt.show()

            if show_wc:
                for i, caps in enumerate(proto_caps):
                    show_wordcloud(caps, f"WordCloud for Cluster {i} of Cluster Head {ch}")
                    
            

        for c in set(clusters.keys()):
            print(f"Cluster {c} holds", len(clusters[c]), "Query Images!")

        print(("Word Cloud and " if show_wc else "")+"Most Common Words for Query KNN-Clusters:")
        for i,caps in enumerate(captions):
            most_common_words = find_most_common_words(caps)
            if show_wc:
                show_wordcloud(caps, f"WordCloud for Query Cluster {i} of Cluster Head {ch}")  
            print("Most Common Words for Query Cluster {i}:") 
            for word, count in most_common_words[:10]:
                print(word, count)


if __name__ == "__main__":
    main()
