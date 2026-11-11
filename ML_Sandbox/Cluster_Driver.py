#%% IMPORT LIBRARIES
# from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ["OMP_NUM_THREADS"] = "2"  # Limit the number of OpenMP threads
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure the local 'data' directory exists44
data_path = r'C:\Users\blain\Documents\MSOL\Python ML Examples\Sandbox\ML_Sandbox'
os.chdir(data_path)
os.makedirs(data_path, exist_ok=True)
print(os.getcwd())

# dataset_type = 'mnist' #'cifar10' #'water_treatment'
# #%% WATER TREATMENT DATA
# if dataset_type == 'water_treatment':
#     # Basic Data Evaluation Block - This is a useful pattern to follow until the data is well-understood
#     filename_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/water-treatment/water-treatment.data'
#     df_train = pd.read_csv(filename_train, parse_dates=True, header = None, encoding='ISO-8859-1')
#     df_train = df_train[~df_train.isin(['?'])] # how to get "NaN" values out of the data.
#     df_train = df_train.dropna()
#     print(df_train.info()) # Print the pandas info block
#     df_train.head()

# #%% MNIST DATASEST
# elif dataset_type == 'mnist':
#     mnist_transform = transforms.Compose([transforms.ToTensor()])
#     mnist_dataset = datasets.MNIST(root=data_path, train=True, transform=mnist_transform, download=True)
#     # train_loader = DataLoader(train_dataset, batch_size=2**8, shuffle=True)
#     #LOAD INTO PANDAS DATAFRAME
#     mnist_images = []
#     mnist_labels = []
#     # img, label = mnist_dataset[0] #sample first 
#     for img, label in mnist_dataset:
#         # convert image to flattened array
#         mnist_images.append(img.numpy().flatten())
#         mnist_labels.append(label)
#     df_train = pd.DataFrame(mnist_images)
#     df_train['labels'] = mnist_labels
#     print(df_train.info())
#     df_train.head()

# #%% CIFAR10 DATASEST
# elif dataset_type == 'cifar10':
#     cifar10_transform = transforms.Compose([transforms.ToTensor()])
#     cifar10_dataset = datasets.CIFAR10(root=data_path, train=True, transform=cifar10_transform, download=True)
#     cifar10_images = []
#     cifar10_labels = []
#     for img, label in cifar10_dataset:
#         cifar10_images.append(img.numpy().flatten())
#         cifar10_labels.append(label)
#     df_train = pd.DataFrame(cifar10_images)
#     df_train['labels'] = cifar10_labels
#     print(df_train.info())
#     df_train.head()

#%%
###########################################################################################
# Code to instantiate Class Starts Below
###########################################################################################
import Dataloader_Tool
import Model_Tool
import Cluster_Tool
import Reconstruct_Tool
import Sample_VAE_Tool
# AE: autoencoder
# AE_contrastive: AE with contrastive learning
# AE_conv: autoencoder convolutional
# VAE: variational autoencoder
# VAE_4NNs: variational autoencoder that uses parameterized NNs in the decoder to distribution of output images from which to sample an output
# VAE_contrastive: VAE with contrastive learning
# VAE_conv: variational autoencoder convolutional
# CVAE: conditional variational autoencoder
# CVAE_contrastive: conditional variational autoencoder with contrastive learning
# AE_conv:
# VAE_conv:
model_type = 'VAE'
dataset_type = 'mnist'#'cifar10'
batch_size = 2**7
num_epochs = 10
latent_size = 256#64
num_latents = 1000
n_clusters = 0 # max 17... number of clusters that are identified by K-Means
cluster_type = 'kmedoids'#'kmeans'
n_components = 2
plt_type = 'px' #'px' #'plt'
# get model and encoding latents
train_dataset, train_loader = Dataloader_Tool.get_dataloader(dataset_type=dataset_type, batch_size=batch_size, plot_flag=False)
model, latent_encodings, labels = Model_Tool.get_model(train_dataset, train_loader, dataset_type, model_type=model_type, num_epochs=num_epochs, latent_size=latent_size, num_latents=num_latents)

# plot reconstructed image
'''This plots the original image and its reconstructed image below it'''
Reconstruct_Tool.plot_reconstruction(dataset_type, train_dataset, model_type, model, 27, sorted=False)

# plot sampled images of VAE models
'''This samples a unit Gaussian for the VAE or CVAE and plots its reconstruction'''
cond_flag = True if model_type == 'CVAE' or model_type == 'CVAE_contrastive' else False 
Sample_VAE_Tool.sample_VAE_image(dataset_type, model, num_images=50, latent_size=latent_size, conditional=cond_flag)
Sample_VAE_Tool.latent_space_interp(dataset_type, model, num_images=10, latent_size=latent_size, conditional=cond_flag)

# put some plotting functions to sample what reconstruction looks like for AE

# pack latent encodings in DataFrame
df_train = pd.DataFrame(latent_encodings)
df_train['labels'] = labels

# Create an instance of the clustering class and make plots
c = Cluster_Tool.cluster(X=df_train, n_clusters=n_clusters)
df_temp_p = c.plot_mds(n_components, cluster_type, plt_type)
df_temp_p = c.plot_pca(n_components, cluster_type, plt_type)
df_temp_t = c.plot_tsne(n_components, cluster_type, plt_type)
df_temp_t = c.plot_umap(n_components, cluster_type, plt_type)

#%% MORE COMPLEX CLUSTERING EXAMPLE
# from __future__ import print_function
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

print(__doc__)

df_train = pd.read_csv(filename_train, parse_dates=True, header = None, encoding='ISO-8859-1')
df_train = df_train[~df_train.isin(['?'])]
df_train = df_train.dropna()
X, Names = prep_dataframe(df_train)
X = preprocessing.MinMaxScaler().fit_transform(X)

range_n_clusters = [3,4,5,6,7,8,10,11,13,30]  # Define how many clusters to create.

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    
    pca = PCA(n_components=3, whiten=True).fit(X)
    X_pca = pca.transform(X)
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X_pca)
    cluster_labels = clusterer.fit_predict(X_pca)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_pca, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

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
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    
    ax2 = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    #print (X[:,3])
    #print (X_pca)    
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], marker='.',  lw=5, alpha=0.7, c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],centers[:, 2],
                marker='o', c="white", alpha=1)
    '''
    for i, txt in enumerate(Names):
        #print (i, txt, X_pca[i][0], X_pca[i][1], X_pca[i][2])
         ax2.text(X_pca[i,0],X_pca[i,1],X_pca[i,2],  '%s' % (txt), size=12, zorder=1, color='k') 
    '''

    #ax2.scatter(centers[:, 0], centers[:, 1],centers[:, 2],
    #           marker='o', c="white", alpha=1, s=200)


    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], c[2], marker='$%d$' % i, alpha=1, s=500)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    ax2.set_zlabel("Feature space for the 3rd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

