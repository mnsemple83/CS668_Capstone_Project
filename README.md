<h1>Spotify Analysis: Grouping songs to build custom playlists</h1>
<h2>Problem Statement</h2>
<p>Spotify has the capability to build custom playlists (also known as Daily Mixes) for its users based on their streaming history. Each of these custom playlists consist of songs that share similarities. Therefore, Spotify must be able to analyze the songs in its library to build these custom playlists.</p>
<h2>Research Question</h2>
<p>How does Spotify use song data to build custom playlists for its users?</p>
<h2>Spotify Tracks Dataset</h2>
<p><b>URL: </b><a href="https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset">https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset</a></p>
<p><b>Description: </b>A public dataset consisting of 114,000 songs from the Spotify library, classified by genre.</p>
<p><b>Year: </b>2023 (as of most recent update).</p>
<p><b>Size: </b>114,000 rows and 20 columns</p>
<ul><b>Parameters (Features):</b>
  <li>track_id</li>
  <li>artists</li>
  <li>album_name</li>
  <li>track_name</li>
  <li>popularity</li>
  <li>duration_ms (converted to duration)</li>
  <li>explicit</li>
  <li>danceability</li>
  <li>energy</li>
  <li>key</li>
  <li>loudness</li>
  <li>mode</li>
  <li>speechiness</li>
  <li>acousticness</li>
  <li>instrumentalness</li>
  <li>liveness</li>
  <li>valence</li>
  <li>tempo</li>
  <li>time_signature</li>
  <li>track_genre</li>
</ul>

<h2>Exploratory Data Analysis</h2>
<ul>
  <li>Songs in the original dataset cover 114 musical genres and 1000 songs for each genre.</li>
  <li>10% of the original dataset sampled (100 songs per genre).</li>
  <li>11 duplicates detected and removed.</li>
  <li>1 observation contained missing values (also removed).</li>
  <li>2,094 outliers were detected, but they were kept since they were the result of natural variations in the musical features and not erroneous data.</li>
  <li>Majority of the songs do not contain explicit content.</li>
  <li>High number rof songs with a poularity score of zero, but they do not make up the majority.</li>
  <li>Most songs are performed in a major key.</li>
  <li>Key correlations include:
    <ul>
      <li>Energy and loudness (strong positive).</li>
      <li>Energy and acousticness (strong negative).</li>
      <li>Acousticness and loudness (strong negative).</li>
    </ul>
    </li>
</ul>
<h2>Methodology</h2>
<ul>
  <li>Extract musical features from the dataset, as well as select non-musical features (popularity, duration, and explicit).</li>
  <li>Apply dimensionality techniques (t-SNE and PCA) to the dataset.</li>
  <li>Use clustering algorithms to group songs: K-Means, K-Means++, and Mini Batch K-Means.</li>
  <li>Evaluate the quality of the clusters for each algorithm to determine which one performed the best.</li>
  <li>Use a Logistic Regression model to cluster new songs based on the already established clusters.</li>
</ul>

<h2>References</h2>
<ul>
  <li><b>[1]</b> H. Tian, H. Cai, J. Wen, S. Li and Y. Li. 2019. A Music Recommendation System Based on Logistic Regression and eXtreme Gradient Boosting. Retrieved October 8, 2023 from <a href='https://ieeexplore.ieee.org/document/8852094'>https://ieeexplore.ieee.org/document/8852094</a></li>
<li><b>[2]</b> Y. Atahan, A. Elbir, A. Enes Keskin, O. Kiraz, B. Kirval and N. Aydin. 2021. Music Genre Classification Using Acoustic Features and Autoencoders. Retrieced October 8, 2023 from <a href='https://ieeexplore.ieee.org/document/9598979'>https://ieeexplore.ieee.org/document/9598979</a></li>
  <li><b>[3]</b> P. N, D. Khanwelkar, H. More, N. Soni, J. Rajani and C. Vaswani. 2022 Analysis of Clustering Algorithms for Music Recommendation. Retrieved October 8, 2023 from <a href='https://ieeexplore.ieee.org/document/9824160'>https://ieeexplore.ieee.org/document/9824160</a></li>
  <li><b>[4]</b> M. Bakhshizadeh, A. Moeini, M. Latifi and M. T. Mahmoudi. 2019. Automated Mood Based Music Playlist Generation by Clustering the Audio Features. Retrieved October 9, 2023 from <a href='https://ieeexplore.ieee.org/document/8965190'>https://ieeexplore.ieee.org/document/8965190</a></li>
  <li><b>[5]</b> H. Wijaya and R. S. Oetama. 2021. Song Similarity Analysis with Clustering Method on Korean Pop Song. Retrieved October 10, 2023 from <a href='https://ieeexplore.ieee.org/document/9617204'>https://ieeexplore.ieee.org/document/9617204</a></li>
  <li><b>[6]</b> R. Sun, J. Zhang, W. Jiang and Y. Hu. 2018. Segmentation of Pop Music Based on Histogram Clustering. Retrieved October 11, 2023 from <a href='https://ieeexplore.ieee.org/document/8633060'>https://ieeexplore.ieee.org/document/8633060</a></li>
  <li><b>[7]</b> H. Han, X. Luo, T. Yang and Y. Shi. 2018. Music Recommendation Based on Feature Similarity. Retrieved October 11, 2023 from <a href='https://ieeexplore.ieee.org/document/8690510'>https://ieeexplore.ieee.org/document/8690510</a></li>
  <li><b>[8]</b> Michelangelo Harris, Brian Liu, Cean Park, Ravi Ramireddy, Gloria Ren, Max Ren, Shangdi Yu, Andrew Daw, and Jamol Pender. 2019. Analyzing the Spotify Top 2000 through a Point Process Lens. Retrieved October 11, 2023 from <a href='https://doi.org/10.48550/arXiv.1910.01445'>https://doi.org/10.48550/arXiv.1910.01445</a></li>
</ul>
