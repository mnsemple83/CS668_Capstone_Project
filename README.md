# Capstone-Project
<h1>Spotify Analysis: Grouping songs to build custom playlists</h1>
<h2>Problem Statement</h2>
<p>Spotify has the capability to build custom playlusts (also known as Daily Mixes) for its users based on their streaming history. Each of these custom playlists consist of songs that share similarities. Therefore, Spotify must be able to analyze the songs in its library to build these custom playlists.</p>
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

<h2>Methodology</h2>
<ul>
  <li>Discard most non-linear features to work with an unlabeled dataset.</li>
  <li>Use dimensionality reduction techniques to determine most important features.</li>
  <li>K-clustering algorithms will be evaluated to determine the best results.</li>
  <li>Best performing K-clustering algorithm will be used to analyze cluster quality.</li>
  <li>Logistic Regression model will try to simulate the addition of new songs to existing playlists.</li>
</ul>
