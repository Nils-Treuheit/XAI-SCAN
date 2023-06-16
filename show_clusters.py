import torch
import yaml
from collections import defaultdict

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

import matplotlib.pyplot as plt
import numpy as np

# Can be between 1 and topk (20)
center_image_index = 15
config_path = "configs/scan/scan_cifar20.yml"
model_path = "results/cifar-20/scan/model.pth.tar"
device = torch.device("mps")



# The images shown are from the cluster of the center_image_index
show_images_for_single_cluster = True

# Show the images from multiple clusters randomly or from prototypes
show_images_multiple_clusters = True
show_from_prototypes = True

# Which prototype clusters to show in effect if show_from_prototypes== True
# Have to be 5
prototype_indices_to_show = [11,12,13,14,15]

# Make predictions for the dataset in order to run the clustering
# Done once the variable can be set to false to save time
making_predictions = False

# Finds captions for the neigbours of the center image 
# Change to true if you change the center_image_index
find_captions = False

# If yoy change from_validation you have to set make_predictions to true
from_validation = True


def main():
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
        
    transforms = get_visualization_transformations(config)

    if from_validation:
        config[
            "topk_neighbors_val_path"
        ] = "results/cifar-20/pretext/topk-val-neighbors.npy"
        dataset = get_val_dataset(config, transforms, to_neighbors_dataset=True)
    else:
        config[
            "topk_neighbors_train_path"
        ] = "results/cifar-20/pretext/topk-train-neighbors.npy"
        dataset = get_train_dataset(config, transforms, to_neighbors_dataset=True)


    # Only needed if we run the predictions again and not from file
    if making_predictions:
        dataloader = get_val_dataloader(config, dataset)

    model = get_model(config)
    model.to(device)
    state_dict = torch.load(model_path, map_location="cpu")

    if config["setup"] in ["simclr", "moco", "selflabel"]:
        model.load_state_dict(state_dict)

    elif config["setup"] == "scan":
        model.load_state_dict(state_dict["model"])

    head = state_dict["head"] if config["setup"] == "scan" else 0

    if making_predictions:
        predictions, features = save_predictions(config, dataloader, model)
        torch.save((predictions, features), "predictions.pt")
    else:
        predictions, features = torch.load("predictions.pt")

    # print("Dataloader Sample Keys:",next(iter(dataloader)).keys())#,"| Dataloader Image Shape:",next(iter(dataloader))["image"].size())
    # print("Predictions keys:",predictions[0].keys(),"| Predictions Shape:",predictions[0]["predictions"].size())
    # print("Neighbours: ", predictions[0]["neighbors"].shape)
    # print("Predictions (clusters): ", predictions[0]["predictions"][1:10])

    # get_clustering_stats(predictions, head)
    
    # Show the distribution of the clusters
    clusters = defaultdict(list)

    for idx, c in enumerate(predictions[0]["predictions"]):
        clusters[c.item()].append(idx)

    for c in set(clusters.keys()):
        print("Cluster", c, ":", len(clusters[c]))


    prototype_indices = get_prototypes(config, predictions[head], features, model)

    print("Top classified prototypes: ", prototype_indices)

    index_of_center = prototype_indices[center_image_index]
    print("Index of center: ", index_of_center)

    images_for_cluster_idx = get_nearest_neighbours_for_image(
        index_of_center, predictions
    )

    print("Number of neighgbours found: ", len(images_for_cluster_idx))

    # target cluster
    cluster = predictions[0]["predictions"][index_of_center]
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
            anchor_idx = prototype_indices[i] if show_from_prototypes else np.random.choice(range(predictions[0]["predictions"].shape[0]))

            neighbour_indicies = predictions[0]["neighbors"][anchor_idx][1:6]
            cluster = predictions[0]["predictions"][anchor_idx]
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
    # show_wordcloud(captions)

    for word, count in most_common_words[:10]:
        print(word, count)

if __name__ == "__main__":
    main()
