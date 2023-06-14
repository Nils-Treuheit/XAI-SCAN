import torch
import yaml
from utils.common_config import (
    get_val_dataset,
    get_val_transformations,
    get_val_dataloader,
    get_model,
    get_train_dataset,
)
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.cluster_visualization import (
    get_nearest_neighbours_for_image,
    get_clustering_stats,
    visualize_cluster,
    save_predictions,
    get_prototypes,
    find_most_common_words,
    show_wordcloud
)


# Can be between 1 and topk (20)
center_image_index = 15
config_path = "configs/scan/scan_cifar20.yml"
model_path = "results/cifar-20/scan/model.pth.tar"
device = torch.device("mps")
show_images = False
making_predictions = False
find_captions = False

def main():
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    config[
        "topk_neighbors_train_path"
    ] = "results/cifar-20/pretext/topk-train-neighbors.npy"

    transforms = get_val_transformations(config)
    dataset = get_train_dataset(config, transforms, to_neighbors_dataset=True)

    # Only needed if we run the predictions again and not from file
    if making_predictions:  dataloader = get_val_dataloader(config, dataset)

    model = get_model(config)
    model.to(device)
    state_dict = torch.load(model_path, map_location="cpu")

    if config["setup"] in ["simclr", "moco", "selflabel"]:
        model.load_state_dict(state_dict)

    elif config["setup"] == "scan":
        model.load_state_dict(state_dict["model"])

    head = state_dict["head"] if config["setup"] == "scan" else 0

    if making_predictions:
        predictions, features = save_predictions(config,dataloader,model)
        torch.save((predictions, features), 'predictions.pt')
    else: predictions, features = torch.load("predictions.pt")


    # print("Dataloader Sample Keys:",next(iter(dataloader)).keys())#,"| Dataloader Image Shape:",next(iter(dataloader))["image"].size())
    # print("Predictions keys:",predictions[0].keys(),"| Predictions Shape:",predictions[0]["predictions"].size())
    # print("Neighbours: ", predictions[0]["neighbors"].shape)
    # print("Predictions (clusters): ", predictions[0]["predictions"][1:10])

    # get_clustering_stats(predictions, head)

    prototype_indices = get_prototypes(config, predictions[head], features, model)
    print("Top classified prototypes: ", prototype_indices)
    print(
        "Prototypes probs shape: \n",
        predictions[0]["probabilities"][prototype_indices].shape,
    )

    index_of_center = prototype_indices[center_image_index]
    print("Index of center:", index_of_center)

    images_for_cluster_idx = get_nearest_neighbours_for_image(
        index_of_center, predictions
    )

    print("Number of neighgbours found: ", len(images_for_cluster_idx))

    # target cluster
    cluster = predictions[0]["predictions"][index_of_center]

    if find_captions:
        captions = visualize_cluster(images_for_cluster_idx, dataset, cluster, show_images)
        torch.save(captions, 'captions.pt')
    else: captions = torch.load('captions.pt')

    most_common_words = find_most_common_words(captions)

    show_wordcloud(captions)
    
    for word, count in most_common_words[:10]:
        print(word, count)

    print("Cluster: ", cluster)


if __name__ == "__main__":
    main()
