# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats

# Import dimensionality reduction algorithms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Import Clustering Algorithms
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

# Import the Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Importing StratifiedKFold for managing inbalanced datasets
from sklearn.model_selection import StratifiedKFold

# Import the statistics module
import statistics

# Import the preprocessing module
from sklearn import preprocessing

# Import standardization technique to normalize the data
from sklearn.preprocessing import StandardScaler

# Import the modules for splitting data for training and testing
from sklearn.model_selection import train_test_split

# Import the GridSearchCV module for parameter optimization
from sklearn.model_selection import GridSearchCV

# Importing metrics for evaluation
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score

# Import the permutation importance module
from sklearn.inspection import permutation_importance

# Import the dataset
df = pd.read_csv('Datasets/spotify_tracks.csv', index_col=0)

# Get the size of the dataset
df.shape

# Display the first 10 records of the dataset
df.head(10)

# Return all column names and their data types
df.dtypes

df.nunique(axis=0)

df.track_genre.unique().tolist()

df.track_genre.value_counts()

# Set the sample rate
rate = 10

# Extract songs from the original dataset
df_subset = df[::rate]

# The size of the new dataset
df_subset.shape

# Get a count of unique values for each feature
df_subset.nunique(axis=0)

# Count the number of tracks for each genre in the new dataset
df_subset.track_genre.value_counts()

# Display descriptive statistics of the dataset
df_subset.describe()

# Count for missing values
print("The following is a summary of missing values:")
print((df_subset.isna().sum().sort_values(ascending=False)))

# Calculate the percentage of missing values for each column
perc_missing = df_subset.isnull().sum() * 100 / len(df_subset)
df_subset_missing_values = pd.DataFrame({'column_name': df_subset.columns,
                                 'percent_missing': perc_missing})
df_subset_missing_values.sort_values('percent_missing', inplace=True, ascending=False)

# Construct a plot to display the percentages of missing values
fig, ax = plt.subplots(figsize=(30, 5))
sns.barplot(
    x='column_name',
    y='percent_missing',
    data=df_subset_missing_values,
    ax=ax
)
plt.xlabel('column_name')
plt.ylabel('percent_missing')
plt.show()

# Find rows with missing data
missing_data = df_subset[df_subset.isnull().any(axis=1)]

missing_data

# Remove rows containing missing data
df_subset = df_subset.dropna()

# Confirm the removal of missing data
# Return the new size of the dataset
df_subset.shape

df_subset[df_subset.duplicated()]

# Count the number of duplicates
len(df_subset[df_subset.duplicated()])

# Remove duplicates
df_subset = df_subset.drop_duplicates()

# Confirm the removal of duplicates
# Return the new size of the dataset
df_subset.shape

df_subset.track_genre.value_counts().tail(10)

# Create a function for converting the explicit feature
def convert_explicit(row):
    
    if row.explicit == True:
        return 1
    else:
        return 0
    return row.explicit

# Convert the explicit feature
df_subset['explicit'] = df_subset.apply(lambda row: convert_explicit(row), axis=1)

# Convert the duration feature from milliseconds to minutes
df_subset['duration_ms'] = df_subset['duration_ms']/60000

# Renaming the duration column
df_subset.rename(columns = {'duration_ms':'duration'}, inplace = True)

df_subset['duration']

# Count songs that contain explicit lyrics/content
# 0 - No
# 1 - Yes
sns.countplot(x = 'explicit', data = df_subset)

# Plot the distribution of songs by popularity
sns.displot(data = df_subset, x='popularity', color='purple', kde=True)

# Plot the distribution of the mode feature
# 0 - Minor
# 1 - Major
sns.countplot(data = df_subset, x = 'mode')

# Plot distribution of the key feature distinguished by mode
# 0 - Minor
# 1 - Major
sns.countplot(data = df_subset, x = 'key', hue = 'mode')

# Continuous features
cont_data = ['duration',
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo']

# Create a dataframe for the continuous features
df_cont = df_subset[cont_data]

# Show the new dataframe
df_cont

# Create a set of colors for each distribution plot
colors = ['lightblue','green','red','palegreen','cyan','tan','wheat','orange','lavender','gold']

# Create a series of subplots for the distribution of each continuous feature
plt.subplots(1, len(df_cont.columns), figsize=(10,14))

for i in range(0, len(df_cont.columns)):
    feature = cont_data[i]
    plt.subplot(5,2,i+1)
    plt.hist(x = df_cont[feature], bins = 30, color = colors[i], ec='black')
    plt.title(feature)

plt.tight_layout()

m_data = ['popularity',
        'duration',
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo']

df_by_genre = df_subset.groupby("track_genre")[m_data].mean()
df_by_genre.reset_index(inplace = True)
df_by_genre.head(10)

# Create a new dataframe for storing the genres with the highest and lowest mean value for each feature
df_mm_genres = pd.DataFrame(columns=['feature','max_genre', 'max_value', 'min_genre', 'min_value'])

# For each feature,
#  get the index for the genre with the highest average value,
#  and the index for the genre with the lowest average value.
# This data will be presented as a summary in a new dataframe.
for m in m_data:
    mx = df_by_genre[m].max() # The maximum value for a given feature
    idx_01 = df_by_genre[m].idxmax() # Index of the row containing the max
    g1 = df_by_genre.iloc[idx_01]['track_genre'] # Genre of the max
    
    mn = df_by_genre[m].min() # The minimum value for a given feature
    idx_02 = df_by_genre[m].idxmin() # Index of the row containing the min
    g2 = df_by_genre.iloc[idx_02]['track_genre'] # Genre of the min
    feat = m # The given feature
    
    # Add to the dataframe
    df_mm_genres.loc[len(df_mm_genres.index)] = [feat, g1, mx, g2, mn]

display(df_mm_genres)

# Extract all continuous features from the dataset.
# Countable features will also be included (explicit, key, mode, popularity, and time_signature)
# Non-linear features are excluded (artists, track_id, track_name, album_name, and track_genre)
data = ['popularity',
        'duration',
        'explicit',
        'danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'time_signature']

# Construct a new dataframe to include only these features
df_new = df_subset[data]

# Create box-and-whisker plots for the features of the new dataframe
plt.subplots(1, len(df_new.columns), figsize=(14,20))

for i in range(0, len(df_new.columns)):
    plt.subplot(3,5,i+1)
    df_new.boxplot(column=data[i], patch_artist=True)

plt.tight_layout()

# Calculate the z-score for all tracks in the new dataframe
z = np.abs(stats.zscore(df_new))
print(z)

# Set threshold for outlier detection
threshold = 3

# Build a list containing the outliers
z_out = z[z > threshold]

# Count the number of outliers
len(z_out)

# Display Correlation Matrix
df_corr = df_new.corr().round(3)
df_corr.style.background_gradient(cmap='coolwarm')

sns.scatterplot(data = df_new, x = 'energy', y = 'loudness')

sns.scatterplot(data = df_new, x = 'acousticness', y = 'energy')

sns.scatterplot(data = df_new, x = 'acousticness', y = 'loudness')

features = ['popularity',
           'duration',
           'explicit',
           'acousticness',
           'danceability',
           'energy',
           'instrumentalness',
           'key',
           'liveness',
           'loudness',
           'mode',
           'tempo',
           'time_signature',
           'valence']

# Create a dataframe to include all the features for our models
df_features = df_new[features]
df_features.head(5)

# Standardize the data
std_features = StandardScaler().fit_transform(df_features)

# Create a dataframe for the standardized values
df_features_std = pd.DataFrame(std_features)
df_features_std.columns = features

# Display the new dataframe with the standardized values
df_features_std.head(10)

# Get the number of rows (observations)
N = len(df_features)

# Optimal perplexity
plex = int(round(N**(1/2),0))
print("Optimal perplexity: " + str(plex))

# Initialize t-SNE
tsne_model = TSNE(perplexity = plex, n_iter = 5000, random_state = 0) # random_state value ensures reproducibility

# Get the t-SNE results
tsne_features = tsne_model.fit_transform(df_features_std)

# Build a dataframe for the t-SNE results
df_tsne = pd.DataFrame(tsne_features, columns = ['tsne0', 'tsne1'])

# Display a sample of the t-SNE dataframe
df_tsne.head(10)

# Plot the resulting clusters
sns.scatterplot(data = df_tsne, x = 'tsne0', y='tsne1')

# Initialize PCA
pca_model = PCA(n_components = 9, random_state=0)

# Get the PCA results
pca_features = pca_model.fit_transform(df_features_std)

# Build a dataframe for the PCA features
df_pca = pd.DataFrame(pca_features, columns = ['pca_01', 
                                               'pca_02', 
                                               'pca_03', 
                                               'pca_04', 
                                               'pca_05', 
                                               'pca_06', 
                                               'pca_07', 
                                               'pca_08',
                                              'pca_09'])

# Display the PCA dataset
df_pca.head(10)

# Creating a table with the explained variance ratio
pca_comps = [f"PCA {i}" for i in range(1, 10, 1)]
scree = pd.DataFrame(list(zip(pca_comps, pca_model.explained_variance_ratio_)), columns=["Component", "EV Ratio"])

# Calculate the cumulative explained variance ratio
cumul_var = []
cumul_var.append(scree['EV Ratio'][0])

for i in range(0, len(scree) - 1):
    c_var = cumul_var[i] + scree['EV Ratio'][i+1]
    cumul_var.append(c_var)

# Update the EVR Table to include the Cumulative Explained Variance Ratio
scree_01 = pd.DataFrame(list(zip(pca_comps, pca_model.explained_variance_ratio_, cumul_var)), columns=["Component", "EV Ratio", "Cumulative EV"])
scree_01

# Plot the explained variance ratio for all the components
sns.lineplot(data = scree_01[['EV Ratio', 'Cumulative EV']]).set(title = "Proportion of Variance")

# Sorting the values of the first principal component by how large each one is
df2 = pd.DataFrame({'PCA':pca_model.components_[0], 'Variable Names':list(df_features.columns)})
df2 = df2.sort_values('PCA', ascending = False)

# Sorting the absolute values of the first principal component by magnitude
df3 = pd.DataFrame(df2)
df3['PCA'] = df3['PCA'].apply(np.absolute)
df3 = df3.sort_values('PCA', ascending = False)

df2.head()

# Sorting the values of the second principal component by how large each one is
df4 = pd.DataFrame({'PCA':pca_model.components_[1], 'Variable Names':list(df_features.columns)})
df4 = df4.sort_values('PCA', ascending = False)

# Sorting the absolute values of the second principal component by magnitude
df5 = pd.DataFrame(df4)
df5['PCA'] = df5['PCA'].apply(np.absolute)
df5 = df5.sort_values('PCA', ascending = False)

df4.head()

# Store the three datasets in new variables
X1 = df_features_std
X2 = df_tsne
X3 = df_pca

# Create a range of cluster values
range_k_values = [2,3,4,5,6,7,8]

# Each model will have a list for the silhouette scores using each dataset.

# K-Means
k_means_silhouette = []
k_means_silhouette_tsne = []
k_means_silhouette_pca = []

# K-Means plus
k_means_plus_silhouette = []
k_means_plus_silhouette_tsne = []
k_means_plus_silhouette_pca = []

# Mini-Batch K-Means
mini_k_means_silhouette = []
mini_k_means_silhouette_tsne = []
mini_k_means_silhouette_pca = []

# Mini-Batch K-Means++
mini_k_means_plus_silhouette = []
mini_k_means_plus_silhouette_tsne = []
mini_k_means_plus_silhouette_pca = []

# Initialize the models
k_means = KMeans(init='random', n_init = 'auto', random_state = 0) # K-Means
k_means_plus = KMeans(init='k-means++', n_init = 'auto', random_state = 0) # K-Means++
mini_k_means = MiniBatchKMeans(init='random', n_init = 'auto', batch_size = 1500, random_state = 0) # Mini Batch K-Means
mini_k_means_plus = MiniBatchKMeans(init='k-means++', n_init = 'auto', batch_size = 1500, random_state = 0) # Mini Batch K-Means++

# Build a function that will perform the following:
# - fit the data to k clusters
# - calculate the silhouette score

# model - The Clustering Algorithm
# X - the dataset
# k_clusters - range of k values
# silhouette_scores - list of silhouette_scores for each value of k
def get_silhouette_scores(model, X, k_clusters, silhouette_scores):
    
    silhouette_scores = [] # empty list
    for k in k_clusters:
        model.n_clusters = k
        model_labels = model.fit_predict(X)
        silhouette_avg = silhouette_score(X, model_labels)
        
        '''
        print("For k clusters = ",
              str(k),
             ", the average silhouette score is ",
              str(silhouette_avg),
             )
        '''
        # Append the silhouette score to the list of scores
        silhouette_scores.append(silhouette_avg)
        
    return silhouette_scores

# Run the get_silhouette_scores function for each algorithm and dataset

# K-Means
k_means_scores = get_silhouette_scores(k_means, X1, range_k_values, 
                                       k_means_silhouette)
k_means_scores_tsne = get_silhouette_scores(k_means, X2, range_k_values,
                                            k_means_silhouette_tsne)
k_means_scores_pca = get_silhouette_scores(k_means, X3, range_k_values, 
                                           k_means_silhouette_pca)

# K-Means++
k_means_plus_scores = get_silhouette_scores(k_means_plus, X1, 
                                            range_k_values, k_means_plus_silhouette)
k_means_plus_scores_tsne = get_silhouette_scores(k_means_plus, X2,
                                                 range_k_values, k_means_plus_silhouette_tsne)
k_means_plus_scores_pca = get_silhouette_scores(k_means_plus, X3, 
                                                range_k_values, k_means_plus_silhouette_pca)

# Mini-Batch K-Means
mini_k_means_scores = get_silhouette_scores(mini_k_means, X1, range_k_values, 
                                                  mini_k_means_silhouette)
mini_k_means_scores_tsne = get_silhouette_scores(mini_k_means, X2, range_k_values, 
                                                  mini_k_means_silhouette_tsne)
mini_k_means_scores_pca = get_silhouette_scores(mini_k_means, X3, range_k_values, 
                                                  mini_k_means_silhouette_pca)

# Mini-Batch K-Means++
mini_k_means_plus_scores = get_silhouette_scores(mini_k_means_plus, X1, range_k_values, 
                                                  mini_k_means_plus_silhouette)
mini_k_means_plus_scores_tsne = get_silhouette_scores(mini_k_means_plus, X2, range_k_values, 
                                                  mini_k_means_plus_silhouette_tsne)
mini_k_means_plus_scores_pca = get_silhouette_scores(mini_k_means_plus, X3, range_k_values, 
                                                  mini_k_means_plus_silhouette_pca)

# Calculate the average silhouette score for each model's performance using each dataset

# K-Means
mean_k_means_scores = statistics.mean(k_means_scores)
mean_k_means_scores_tsne = statistics.mean(k_means_scores_tsne)
mean_k_means_scores_pca = statistics.mean(k_means_scores_pca)

# K-Means++
mean_k_means_plus_scores = statistics.mean(k_means_plus_scores)
mean_k_means_plus_scores_tsne = statistics.mean(k_means_plus_scores_tsne)
mean_k_means_plus_scores_pca = statistics.mean(k_means_plus_scores_pca)

# Mini-Batch K-Means
mean_mini_k_means_scores = statistics.mean(mini_k_means_scores)
mean_mini_k_means_scores_tsne = statistics.mean(mini_k_means_scores_tsne)
mean_mini_k_means_scores_pca = statistics.mean(mini_k_means_scores_pca)

# Mini-Batch K-Means++
mean_mini_k_means_plus_scores = statistics.mean(mini_k_means_plus_scores)
mean_mini_k_means_plus_scores_tsne = statistics.mean(mini_k_means_plus_scores_tsne)
mean_mini_k_means_plus_scores_pca = statistics.mean(mini_k_means_plus_scores_pca)

# Construct a plot to display the mean silhouette scores for each algorithm and dataset

# Set the width of the bar
barWidth = 0.25
fig = plt.subplots(figsize =(16, 8)) 

# The height of the bars will represent the mean silhouette scores.

# Original
original = [mean_k_means_scores,
            mean_k_means_plus_scores,
            mean_mini_k_means_scores,
            mean_mini_k_means_plus_scores] 

# t-SNE
tsne = [mean_k_means_scores_tsne,
        mean_k_means_plus_scores_tsne,
        mean_mini_k_means_scores_tsne,
        mean_mini_k_means_plus_scores_tsne]

# PCA
pca = [mean_k_means_scores_pca,
       mean_k_means_plus_scores_pca,
       mean_mini_k_means_scores_pca,
       mean_mini_k_means_plus_scores_pca] 

# Set position of bar on X axis 
br1 = np.arange(len(original)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

# Make the plot

# Original dataset
plt.bar(br1, original, color ='lightblue', width = barWidth, 
        edgecolor ='grey', label ='Original') 

# t-SNE dataset
plt.bar(br2, tsne, color ='linen', width = barWidth, 
        edgecolor ='grey', label ='t-SNE')

# PCA dataset
plt.bar(br3, pca, color ='palegreen', width = barWidth, 
        edgecolor ='grey', label ='PCA') 

# Adding Xticks 
plt.xlabel('Models', fontweight ='bold', fontsize = 15) 
plt.ylabel('Mean Silhouette Score', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(original))], 
           ['K-Means', 'K-Means++', 'Mini-Batch K-Means', 'Mini-Batch K-Means++'])

plt.legend(loc = 'upper right')
plt.show() 

# Set up the columns and row data
d1 = {'Model': ['K-Means','K-Means++','Mini-Batch K-Means', 'Mini-Batch K-Means++'],
         'Dataset': ['t-SNE','t-SNE','t-SNE','t-SNE'],
         'Mean Silhouette Score': [mean_k_means_scores_tsne,
                                  mean_k_means_plus_scores_tsne,
                                  mean_mini_k_means_scores_tsne,
                                  mean_mini_k_means_plus_scores_tsne]}

# Create the dataframe for the t-SNE dataset
df_tsne_scores = pd.DataFrame(d1)
df_tsne_scores

# Create lists to store the DBI scores for each algorithm
k_means_dbi_scores = []
k_means_plus_dbi_scores = []
mini_k_means_dbi_scores = []
mini_k_means_plus_dbi_scores = []

# Build a function for calculating the DBI for each cluster
def get_DBI(model, X, k_clusters, dbi_scores):
    
    dbi_scores = [] # empty list
    for k in k_clusters:
        model.n_clusters = k
        model_labels = model.fit_predict(X)
        db_index = davies_bouldin_score(X, model_labels)
        
        # Append the dbi score to the list of scores
        dbi_scores.append(db_index)
        
    return dbi_scores    

# Calculate the DBI scores
dbi_scores_01 = get_DBI(k_means,X2,range_k_values,k_means_dbi_scores) # K-Means
dbi_scores_02 = get_DBI(k_means_plus,X2,range_k_values,k_means_plus_dbi_scores) # K-Means++
dbi_scores_03 = get_DBI(mini_k_means,X2,range_k_values,mini_k_means_dbi_scores) # Mini-Batch K-Means
dbi_scores_04 = get_DBI(mini_k_means_plus,X2,range_k_values,mini_k_means_plus_dbi_scores) # Mini-Batch K-Means++

# Calcluate the mean DBI scores
mean_dbi_01 = statistics.mean(dbi_scores_01) # K-Means
mean_dbi_02 = statistics.mean(dbi_scores_02) # K-Means++
mean_dbi_03 = statistics.mean(dbi_scores_03) # Mini-Batch K-Means
mean_dbi_04 = statistics.mean(dbi_scores_04) # Mini-Batch K-Means++

# Construct a plot to display the mean DBI scores

data = {'K-Means': mean_dbi_01,
        'K-Means++': mean_dbi_02,
        'Mini-Batch K-Means': mean_dbi_03,
        'Mini-Batch K-Means++': mean_dbi_04}

models = list(data.keys())
scores = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# Creating the bar plot
plt.bar(models, scores, color ='turquoise', 
        width = 0.4)
 
plt.xlabel("Models")
plt.ylabel("Mean DBI Score")
plt.title("Mean Davies-Bouldin Index Scores for the K-clustering algorithms")
plt.show()

# Build a dataframe for comparing the mean Davies-Bouldin Index scores
#  with the mean Silhouette Scores.

# The dictionary
data_01 = {'Model': list(data.keys()), 
           'Mean Silhouette Score': [mean_k_means_scores_tsne,
                                  mean_k_means_plus_scores_tsne,
                                  mean_mini_k_means_scores_tsne,
                                  mean_mini_k_means_plus_scores_tsne], 
           'Mean DBI Score': (data.values())}

# The dataframe
df_mean_scores = pd.DataFrame(data_01)
df_mean_scores

for k_clusters in range_k_values:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (k_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X2) + (k_clusters + 1) * 10])

    # Initialize the model with k_clusters value and a random generator
    # seed of 0 for reproducibility.
    model = KMeans(n_clusters = k_clusters, init = "random", n_init = "auto", random_state = 0)
    model_labels = model.fit_predict(X2)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X2, model_labels)
    print(
        "For k_clusters =",
        k_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X2, model_labels)

    y_lower = 10
    for i in range(k_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[model_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor = color,
            edgecolor = color,
            alpha = 0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x = silhouette_avg, color = "red", linestyle = "--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(model_labels.astype(float) / k_clusters)
    ax2.scatter(
        X2['tsne0'], X2['tsne1'], marker = ".", s = 30, lw = 0, alpha = 0.7, c = colors, edgecolor = "k"
    )

    # Labeling the clusters
    centers = model.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker = "o",
        c = "white",
        alpha = 1,
        s = 200,
        edgecolor = "k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker = "$%d$" % i, alpha = 1, s = 50, edgecolor = "k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st t-SNE component")
    ax2.set_ylabel("Feature space for the 2nd t-SNE component")

    plt.suptitle(
        "Silhouette analysis for K-Means clustering on t-SNE data with k_clusters = %d"
        % k_clusters,
        fontsize = 14,
        fontweight = "bold",
    )

plt.show()

# Initialize the K-Means algorithm with the value of k = 3
# Then, fit the t-SNE dataset to get the labels
k3_model = KMeans(n_clusters = 3, init = "random", n_init = "auto", random_state = 0)
k3_labels = k3_model.fit_predict(X2)

# Declare a list to be converted to a column
cluster_k3 = list(k3_labels) # For k = 3 dataset

# Create copies of the sampled dataset
df_k3_subset = df_subset

# Add the cluster column to the new dataframe
df_k3_subset['cluster'] = cluster_k3

# View the result for the K = 3 model
df_k3_subset.head(10)

# Create a list of all the features to be observed
obs_features = [
        'popularity',
        'duration',
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo']

# Build a dataframe for the mean values of each feature by cluster

df_k3_averages = df_k3_subset.groupby("cluster")[obs_features].mean()
df_k3_averages.reset_index(inplace = True)
df_k3_averages.head(10)

# Create a dataframe for displaying the clusters containing the highest and lowest mean values for each feature
df_mm_cluster = pd.DataFrame(columns=['feature','max_cluster', 'max_value', 'min_cluster', 'min_value'])

# Fill the dataframe with the maximum and minimum values for each feature,
#  and the clusters that contain those respective values.
for fe in obs_features:
    mx_ft = df_k3_averages[fe].max() # The maximum value for a given feature
    idx_01 = df_k3_averages[fe].idxmax() # Index of the row containing the max
    c_max = str(int(df_k3_averages.iloc[idx_01]['cluster'])) # Cluster of the max
    
    mn_ft = df_k3_averages[fe].min() # The minimum value for a given feature
    idx_02 = df_k3_averages[fe].idxmin() # Index of the row contaiing the min
    c_min = str(int(df_k3_averages.iloc[idx_02]['cluster'])) # Cluster of the min
    
    feat = fe # The given feature
    
    # Add to the dataframe
    df_mm_cluster.loc[len(df_mm_cluster.index)] = [feat, c_max, mx_ft, c_min, mn_ft]

# Display the dataframe
display(df_mm_cluster)

# Set up the X and y variables
X4 = df_k3_subset[obs_features] # Features
y1 = df_k3_subset.cluster # Target

# Return a count of songs for each clsuter
pd.DataFrame(y1.value_counts())

# train - training data
# test - testing data
# fold_no - fold number
# train_X - feature values for the training data used for each fold
# train_y - target values for the training data used for each fold
# test_X - feature values for the testing data used for each fold
# test_y - target values for the testing data used for each fold
# preds - predictions
# model_0 - the model to be trained
# scores - contains a list of the F1 scores for each fold
def train_model(train, test, fold_no, train_X, train_y, test_X, test_y, preds, model_0, scores):
    # Set up features and targets
    X0 = obs_features # features
    y0 = ['cluster'] # target
    
    # Scale features for both training and testing sets
    scaler = preprocessing.StandardScaler()
    
    # Initialize features and targets for training and testing
    X0_train = scaler.fit_transform(train[X0])
    y0_train = train[y0]
    X0_test = scaler.fit_transform(test[X0])
    y0_test = test[y0]
    
    # Fit the training data to the model
    model_0.fit(X0_train,y0_train.values.ravel())
    
    # Make the predictions
    y0_pred = model_0.predict(X0_test)
    
    # Append training and testing data to lists for storage
    # This data will be used to determine the best folds
    train_X.append(X0_train)
    train_y.append(y0_train)
    test_X.append(X0_test)
    test_y.append(y0_test)
    
    # Append predictions to a list
    preds.append(y0_pred)
    
    # Each model will have its own list of F1 scores to evaluate their overall performance
    scores.append(round(metrics.f1_score(y0_test,y0_pred,average='weighted'), 3))
    
    print('Fold',str(fold_no),'F1 Score:',round(metrics.f1_score(y0_test,y0_pred,average='weighted'),3))

# The predictions for each fold
fold_preds_k3 = []
# The F1 Scores of each fold
fold_scores_k3 = []

# List of training sets used for each fold
X_train_sets_k3 = []
y_train_sets_k3 = []

# List of testing sets used for each fold
# These will be used for evaluation purposes
X_test_sets_k3 = []
y_test_sets_k3 = []

log_k3 = LogisticRegression(random_state=0, multi_class='multinomial')

# Initialize the Stratified K-fold algorithm
# The number of splits will be set to 10
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

fold_no = 1
for train_idx, test_idx in skf.split(df_k3_subset, y1):
    train_k3 = df_k3_subset.iloc[train_idx,:]
    test_k3 = df_k3_subset.iloc[test_idx,:]
    train_model(train_k3, test_k3, fold_no, X_train_sets_k3, y_train_sets_k3, 
                X_test_sets_k3, y_test_sets_k3, fold_preds_k3, log_k3, fold_scores_k3)
    fold_no += 1
    
# Print the results
print('\nMaximum F1 Score that can be obtained from this model is:',
      max(fold_scores_k3))
print('\nMinimum F1 Score:',
      min(fold_scores_k3))
print('\nOverall F1 Score:',
      round(statistics.mean(fold_scores_k3),3))

best_index_k3 = fold_scores_k3.index(max(fold_scores_k3)) # The index
best_preds_k3 = fold_preds_k3[best_index_k3] # The predictions

# Training data
best_X_train_k3 = X_train_sets_k3[best_index_k3]
best_y_train_k3 = y_train_sets_k3[best_index_k3]

# Testing data
best_X_test_k3 = X_test_sets_k3[best_index_k3]
best_y_test_k3 = y_test_sets_k3[best_index_k3]

# Set up the hyperparameters
c_space = np.logspace(-5, 8, 15)
iterations = np.arange(100,1000,100)
log_params = {'C': c_space, 'max_iter': iterations}

# Re-initialize the Logistic Regression model
log_k3 = LogisticRegression(random_state=0, multi_class='multinomial')

# Initialize the model using the hyperparameters
log_k3_gs = GridSearchCV(log_k3, log_params)

# Store the name of the model for easy reference
model_name = type(log_k3_gs).__name__

# Train the model
log_k3_gs.fit(best_X_train_k3, best_y_train_k3.values.ravel())
    
# Training set predictions
train_preds_k3 = log_k3_gs.predict(best_X_train_k3)
    
# Testing set predictions
test_preds_k3 = log_k3_gs.predict(best_X_test_k3)
    
# Results
train_k3_f1_score = round(metrics.f1_score(best_y_train_k3, train_preds_k3, average='weighted',zero_division=0), 3)
test_k3_f1_score = round(metrics.f1_score(best_y_test_k3, test_preds_k3, average='weighted', zero_division=0), 3)
    
print("\nResults for the " + str(model_name) + ": ")
print("Training: " + str(train_k3_f1_score) + ", Testing: " + str(test_k3_f1_score))

result = permutation_importance(log_k3_gs, best_X_test_k3, best_y_test_k3, n_repeats=10, random_state=42)


feature_importance = pd.DataFrame({'Feature': X4.columns,
                                   'Importance': result.importances_mean,
                                   'Standard Deviation': result.importances_std})
feature_importance = feature_importance.sort_values('Importance', ascending=True)


ax = feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), yerr='Standard Deviation', capsize=4)
ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importance with Standard Deviation (K = 3 Dataset)')

print("Best Logistic Regression parameters: {}".format(log_k3_gs.best_params_))

# Create cluster labels
clus_labels_k3 = list(y1.unique())
clus_labels_k3.sort()

# Create the confusion matrix
conf_matrix_k3 = metrics.confusion_matrix(best_y_test_k3,test_preds_k3)

# Build the display for the confusion matrix
mat_display_k3 = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix_k3, display_labels = clus_labels_k3)

# Get the F1 Score for the Logistic Regression model
log_k3_f1_score = round(metrics.f1_score(best_y_test_k3, test_preds_k3, average='weighted', zero_division=0),3)

# Display the confusion matrix
mat_display_k3.plot()
plt.ylabel('Actual Cluster');
plt.xlabel('Predicted Cluster');
all_sample_title = 'Model F1 Score (K = 3 Dataset): {0}'.format(log_k3_f1_score)
plt.title(all_sample_title, size = 12)
plt.show()

clf_report_k3 = metrics.classification_report(best_y_test_k3, test_preds_k3, zero_division=0)
print(clf_report_k3)

# Initialize the K-Means algorithm with the value of k = 4
# Then, fit the t-SNE dataset to get the labels
k4_model = KMeans(n_clusters = 4, init = "random", n_init = "auto", random_state = 0)
k4_labels = k4_model.fit_predict(X2)

# Declare a list to be converted to a column
cluster_k4 = list(k4_labels) # For K = 4 dataset

# Create copies of the sampled dataset
df_k4_subset = df_subset

# Add the cluster column to the new dataframe
df_k4_subset['cluster'] = cluster_k4

# View the result for the K = 4 model
df_k4_subset.head(10)

# Build a dataframe for the mean values of each feature by cluster

df_k4_averages = df_k4_subset.groupby("cluster")[obs_features].mean()
df_k4_averages.reset_index(inplace = True)
df_k4_averages.head(10)

# Create a dataframe for displaying the clusters containing the highest and lowest mean values for each feature
df_mm_cluster_02 = pd.DataFrame(columns=['feature','max_cluster', 'max_value', 'min_cluster', 'min_value'])

# Fill the dataframe with the maximum and minimum values for each feature,
#  and the clusters that contain those respective values.
for fe in obs_features:
    mx_ft = df_k4_averages[fe].max() # The maximum value for a given feature
    idx_01 = df_k4_averages[fe].idxmax() # Index of the row containing the max
    c_max = str(int(df_k4_averages.iloc[idx_01]['cluster'])) # Cluster of the max
    
    mn_ft = df_k4_averages[fe].min() # The minimum value for a given feature
    idx_02 = df_k4_averages[fe].idxmin() # Index of the row contaiing the min
    c_min = str(int(df_k4_averages.iloc[idx_02]['cluster'])) # Cluster of the min
    
    feat = fe # The given feature
    
    # Add to the dataframe
    df_mm_cluster_02.loc[len(df_mm_cluster_02.index)] = [feat, c_max, mx_ft, c_min, mn_ft]

# Display the dataframe
display(df_mm_cluster_02)

# Set up the X and y variables
X5 = df_k4_subset[obs_features] # Features
y2 = df_k4_subset.cluster # Target
pd.DataFrame(y2.value_counts())

# The predictions for each fold
fold_preds_k4 = []
# The F1 Scores of each fold
fold_scores_k4 = []

# List of training sets used for each fold
X_train_sets_k4 = []
y_train_sets_k4 = []

# List of testing sets used for each fold
# These will be used for evaluation purposes
X_test_sets_k4 = []
y_test_sets_k4 = []

log_k4 = LogisticRegression(random_state=0, multi_class='multinomial')

# Initialize the Stratified K-fold algorithm
# The number of splits will be set to 10
skf_02 = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

fold_no = 1
for train_idx, test_idx in skf_02.split(df_k4_subset, y2):
    train_k4 = df_k4_subset.iloc[train_idx,:]
    test_k4 = df_k4_subset.iloc[test_idx,:]
    train_model(train_k4, test_k4, fold_no, X_train_sets_k4, y_train_sets_k4, 
                X_test_sets_k4, y_test_sets_k4, fold_preds_k4, log_k4, fold_scores_k4)
    fold_no += 1
    
# Print the results
print('\nMaximum F1 Score that can be obtained from this model is:',
      max(fold_scores_k4))
print('\nMinimum F1 Score:',
      min(fold_scores_k4))
print('\nOverall F1 Score:',
      round(statistics.mean(fold_scores_k4),3))

best_index_k4 = fold_scores_k4.index(max(fold_scores_k4)) # The index
best_preds_k4 = fold_preds_k4[best_index_k4] # The predictions

# Training data
best_X_train_k4 = X_train_sets_k4[best_index_k4]
best_y_train_k4 = y_train_sets_k4[best_index_k4]

# Testing data
best_X_test_k4 = X_test_sets_k4[best_index_k4]
best_y_test_k4 = y_test_sets_k4[best_index_k4]

# Re-initialize the Logistic Regression model
log_k4 = LogisticRegression(random_state=0, multi_class='multinomial')

# Initialize the model using the hyperparameters
log_k4_gs = GridSearchCV(log_k4, log_params)

# Store the name of the model for easy reference
model_name_02 = type(log_k4_gs).__name__

# Train the model
log_k4_gs.fit(best_X_train_k4, best_y_train_k4.values.ravel())
    
# Training set predictions
train_preds_k4 = log_k4_gs.predict(best_X_train_k4)
    
# Testing set predictions
test_preds_k4 = log_k4_gs.predict(best_X_test_k4)
    
# Results
train_k4_f1_score = round(metrics.f1_score(best_y_train_k4, train_preds_k4, average='weighted',zero_division=0), 3)
test_k4_f1_score = round(metrics.f1_score(best_y_test_k4, test_preds_k4, average='weighted', zero_division=0), 3)
    
print("\nResults for the " + str(model_name_02) + ": ")
print("Training: " + str(train_k4_f1_score) + ", Testing: " + str(test_k4_f1_score))

print("Best Logistic Regression parameters: {}".format(log_k4_gs.best_params_))

result = permutation_importance(log_k4_gs, best_X_test_k4, best_y_test_k4, n_repeats=10, random_state=42)


feature_importance = pd.DataFrame({'Feature': X5.columns,
                                   'Importance': result.importances_mean,
                                   'Standard Deviation': result.importances_std})
feature_importance = feature_importance.sort_values('Importance', ascending=True)


ax = feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6), yerr='Standard Deviation', capsize=4)
ax.set_xlabel('Permutation Importance')
ax.set_title('Permutation Importance with Standard Deviation (K = 4 Dataset)')

# Create cluster labels
clus_labels_k4 = list(y2.unique())
clus_labels_k4.sort()

# Create the confusion matrix
conf_matrix_k4 = metrics.confusion_matrix(best_y_test_k4,test_preds_k4)

# Build the display for the confusion matrix
mat_display_k4 = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix_k4, display_labels = clus_labels_k4)

# Get the F1 Score for the Logistic Regression model
log_k4_f1_score = round(metrics.f1_score(best_y_test_k4, test_preds_k4, average='weighted', zero_division=0),3)

# Display the confusion matrix
mat_display_k4.plot()
plt.ylabel('Actual Cluster');
plt.xlabel('Predicted Cluster');
all_sample_title = 'Model F1 Score (K = 4 Dataset): {0}'.format(log_k4_f1_score)
plt.title(all_sample_title, size = 12)
plt.show()

clf_report_k4 = metrics.classification_report(best_y_test_k4, test_preds_k4, zero_division=0)
print(clf_report_k4)



