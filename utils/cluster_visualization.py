import torch
import torch
from utils.evaluate_utils import get_predictions, hungarian_evaluate
import time
from transformers import pipeline
from collections import Counter
import nltk
from nltk.corpus import stopwords
import tqdm
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from data.custom_dataset import NeighborsDataset
from sampleDataSet import SampleDataSet
from PIL import Image 

# nltk.download('stopwords')


@torch.no_grad()
def get_prototypes(predictions, features, topk=20):
    import torch.nn.functional as F

    # Get topk most certain indices and pred labels
    print("Get topk...")
    # Filter for differenr prediction["targets"]
    probs = predictions["probabilities"]
    n_classes = probs.shape[1]
    dims = features.shape[1]
    max_probs, pred_labels = torch.max(probs, dim=1)
    indices = torch.zeros((n_classes, topk))

    for pred_id in range(n_classes):
        probs_copy = max_probs.clone()
        mask_out = ~(pred_labels == pred_id)
        probs_copy[mask_out] = -1
        conf_vals, conf_idx = torch.topk(probs_copy, k=topk, largest=True, sorted=True)
        indices[pred_id, :] = conf_idx

    # Get corresponding features
    selected_features = torch.index_select(
        features, dim=0, index=indices.view(-1).long()
    )
    selected_features = selected_features.unsqueeze(1).view(n_classes, -1, dims)

    # Get mean feature per class
    mean_features = torch.mean(selected_features, dim=1)

    # Get min distance wrt to mean
    diff_features = selected_features - mean_features.unsqueeze(1)
    diff_norm = torch.norm(diff_features, 2, dim=2)

    # Get final indices
    _, best_indices = torch.min(diff_norm, dim=1)
    one_hot = F.one_hot(best_indices.long(), indices.size(1)).byte()

    # New make vector from 1 and 0 to boolean
    one_hot = torch.gt(one_hot, 0)
    proto_indices = torch.masked_select(indices.view(-1), one_hot.view(-1))
    proto_indices = proto_indices.int().tolist()
    return proto_indices


def save_predictions(config, dataloader, model):
    with torch.no_grad():
        print("Getting predictions...")
        start = time.time()
        predictions, features = get_predictions(
            config, dataloader, model, return_features=True
        )
        end = time.time()
        print("Time to make predictions: ", end - start)
        return predictions, features


def get_nearest_neighbours_for_image(index_of_center, predictions, complete_preds = None):
    if not complete_preds:
        complete_preds = predictions

    center_image_neighbours = predictions["neighbors"][index_of_center]
    print("center nei: ", center_image_neighbours)
    neighbour_indexes = []
    for image_idx in center_image_neighbours:
        neigbours_idx = complete_preds["neighbors"][image_idx]

        for idx in neigbours_idx:
            n_list = complete_preds["neighbors"][idx]
            n_list = n_list.tolist()
            neighbour_indexes = neighbour_indexes + n_list

    combined = set(neighbour_indexes)
    return combined


def visualize_cluster(images_for_cluster_idx, dataset, cluster, show_images=True):
    images = []
    captions = []
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    for idx in tqdm(
        images_for_cluster_idx,
        total=len(images_for_cluster_idx),
        desc="Getting captions...",
    ):
        img = dataset.dataset.get_image(idx) if(isinstance(dataset,NeighborsDataset)) else dataset.get_image(idx)
        image = Image.fromarray(img)

        caption = captioner(images=image, max_new_tokens=10)[0]["generated_text"]
        filtered_caption = remove_stopwords(caption)

        if show_images:
            # image = np.transpose(image, (1, 2, 0))
            plt.imshow(image)
            plt.title(f"Cluster: {cluster}, Caption: {caption}")
            plt.show()
        captions.append(filtered_caption)
        images.append(image)

    return captions


def get_clustering_stats(predictions, head):
    all_clusters = torch.unique(predictions[0]["targets"])
    all_clusters_str = [str(value) for value in all_clusters.tolist()]

    clustering_stats = hungarian_evaluate(
        head, predictions, all_clusters_str, compute_confusion_matrix=True
    )
    print("Clustering stats: ", clustering_stats)


def find_most_common_words(sentences):
    word_counts = Counter()
    for sentence in sentences:
        words = sentence.split()
        word_counts.update(words)

    most_common_words = word_counts.most_common(10)
    return most_common_words


def remove_stopwords(sentence):
    stop_words: set = set(stopwords.words("english"))
    words: list = sentence.split()
    filtered_words: list = [word for word in words if word.lower() not in stop_words]
    filtered_sentence = " ".join(filtered_words)
    return filtered_sentence


def show_wordcloud(captions, title = None):
    single_string: str = ""
    for capture in tqdm(captions, total=len(captions), desc="Building word cloud"):
        single_string += capture + "\n"

    # Creating word_cloud with text as argument in .generate() method
    word_cloud = WordCloud(collocations=False, background_color="white").generate(
        single_string
    )
    # Display the generated Word Cloud
    plt.imshow(word_cloud, interpolation="bilinear")
    if title!=None: plt.suptitle(title)
    plt.axis("off")
    plt.show()


def get_caption(image ,max_new_tokens=4):
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    caption = captioner(images=image, max_new_tokens=max_new_tokens)[0][
        "generated_text"
    ]
    filtered_caption = remove_stopwords(caption)
    return filtered_caption
