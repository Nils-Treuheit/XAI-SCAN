from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
X, y = make_blobs(
    n_samples=50001,
    n_features=512,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)


def plot_silhouette_analysis(features, labels, cluster_heads, unique_values_count, unique_values):
    for k in range(len(cluster_heads)):
        n_clusters = unique_values_count[cluster_heads[k]]
        if n_clusters > 1:
            cluster_labels = labels[cluster_heads[k]]

            # Create a subplot with 1 row and 2 columns
            fig, (ax1) = plt.subplots(1)
            fig.set_size_inches(18, 7)
            silhouette_avg = silhouette_score(features, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(
                features, cluster_labels)

            y_lower = 10
            for i in unique_values[cluster_heads[k]]:

                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(
                (np.array(cluster_labels)).astype(float) / n_clusters)

            plt.suptitle(
                "Silhouette analysis for CIFAR20 image clustering using SCAN n_clusters = %d"
                % cluster_heads[k],
                fontsize=14,
                fontweight="bold",
            )
    plt.show()


def extrinsic_evaluation():
    x = [5, 20, 100]
    nmi = [30, 50, 30]
    ari = [10, 30, 20]

    plt.figure(num=3, figsize=(8, 5))

    plt.plot(x, nmi,
        color='blue',
        linewidth=1.0, label='Normalized Mutual Info Score')

    plt.plot(x, ari,
        color='green',
        linewidth=1.0, label='Adjusted Rand Score')
    plt.ylim((0, 100))

    plt.scatter(x, nmi, color='blue')
    plt.scatter(x, ari, color='green')

    plt.xlabel("Number of neighbours - K")
    plt.ylabel("Metrics")
    plt.title("Extrinsic Performance Metrics")
    plt.legend()

    return plt.show()


def intrinsic_evaluation():
    x = [5,20,100, 300, 500]
    k20 = [0.05, 0.08 , 0.1, 0.08, 0.06]
    k5 = [0.06, 0.07, 0.08, 0.06, 0.05]
    k100 = [0.05, 0.06, 0.07, 0.05, 0.04]

    plt.figure(num=3, figsize=(8, 5))
      
    plt.plot(x, k5,
         color='brown', label='K = 5')

    plt.plot(x, k20, 
         color='blue',  
         linewidth=1.0, label='K = 20')

    plt.plot(x, k100, 
         color='green',  
         linewidth=1.0, label='K = 100')
  
    plt.ylim((0, 0.3))  

    plt.scatter(x, k5, color='brown')
    plt.scatter(x, k20, color='blue')
    plt.scatter(x, k100, color='green')

    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Intrinsic Performance Metrics")
    plt.legend()


    return plt.show()


if __name__ == "__main__":
    main()
