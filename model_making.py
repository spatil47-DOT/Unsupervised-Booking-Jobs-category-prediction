"""
Author- Suraj Prakash Patil
Date- 16/05/2024
Text Preprocessing, model making and madel saving for Unsupervised Learning
"""


import warnings
warnings.simplefilter("ignore", FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import completeness_score
from sklearn.cluster import KMeans
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


"""
Here we import Files for the Jobs dataset
"""
df = pd.read_csv('Booking_Jobs_All_220218/Booking_Jobs_All_220218.csv')
train=df
df.head(5)
print(train.nunique())

"""
1) Text Preprocessing 
Here we perform Cleaning & Vectorization on raw data for the Jobs dataset. 
Here we obtain the Vector space representation  of the dataset.
"""  

tfidf_vect = TfidfVectorizer(sublinear_tf=True,
                            max_df=0.95,
                            min_df=round(len(train["Job Description"])*0.01),
                            stop_words="english")

tfidf_vect.fit(train["Job Description"])
tfidf_train = tfidf_vect.transform(train["Job Description"])
print(tfidf_train.shape)

#############################################################################
'''
2)  Selection of Number of natural clusters
'''
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(2, 15)
X=tfidf_train


# Initializing lists to store inertia and silhouette scores
inertia = []
silhouette_scores = []

# For  optimal number of clusters using the elbow method
for n_clusters in range(2, 15):  # Trying different numbers of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# For Plotting the elbow method graph
plt.plot(range(2, 15), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# For optimal number of clusters using silhouette analysis
for n_clusters in range(2, 15):  # For different numbers of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# For Plotting the silhouette scores
plt.plot(range(2, 15), silhouette_scores, marker='o')
plt.title('Silhouette Analysis for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

#############################################################################
"""
3) Here we perform KMeans Clustering on vectorised data for the Jobs dataset. 
From 2) we deduce numbers of clusters  
"""  
kmeans = KMeans(n_clusters= 14, random_state=42)
kmeans.fit(tfidf_train)
labels = kmeans.labels_
train_labels = train["Team"].factorize()[0]
print(completeness_score(train_labels, labels))


"""
Here we save the model to the drive for the Jobs dataset
""" 
with open("model_jobs_1.pkl", "wb") as f:
    pickle.dump(kmeans, f)


"""
Here we test the model with sample data to check code output
""" 
test_text= train["Job Description"][90]
print(kmeans.predict(tfidf_vect.transform([test_text])))



#############################################################################
'''
4)	Identify key words from each cluster using WordClouds
'''


# Step 3: Extract cluster labels
train['Cluster'] = kmeans.labels_
num_clusters = n_clusters  # Choose the number of clusters

# Step 4: Generate word clouds for each cluster
for cluster_label in range(num_clusters):
    cluster_text = ' '.join(train[train['Cluster'] == cluster_label]['Job Description'])
    wordcloud = WordCloud(background_color='white').generate(cluster_text)

    # Display the word cloud
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for Cluster {cluster_label}')
    plt.axis('off')
    plt.show()


