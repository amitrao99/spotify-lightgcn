"""
This script performs preprocessing on the full Spotify Million Playlist Dataset, which can be downloaded
from https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
(although you will need to make an account).

Specifically, it first uses the SNAP library to calculate the K-core subgraph. It then performs some
re-indexing so that the remaining nodes have sequential indexing. Then it converts the graph to the Data
format from PyTorch Geometric and saves to a file. Along the way, it also stores some information about the
songs/playlists in the resulting dataset, and saves these to a file. That info will be used later if you want
to perform some analysis on the resulting embeddings, etc.

Outputs (all will be saved in save_dir):
  data_object.pt: resulting graph in PyTorch Geometric's Data format. Playlists will have indices from 0...num_playlists-1,
     and songs will have indices from num_playlists...num_nodes-1
  playlist_info.json: JSON file mapping playlist ID to some information about that playlist. Can be used later for analysis
  song_info.json: JSON file mapping song ID to some information about that song. Can be used later for analysis
"""

import json
import random
import numpy as np
import os
import snap
from tqdm import tqdm
import torch
from torch_geometric.data import Data

random.seed(5)
np.random.seed(5)


##### SET THESE VALUES BEFORE RUNNING
data_dir = './data' # path to Spotify dataset files
NUM_FILES_TO_USE = 30 # will create dataset based on the first NUM_FILES_TO_USE files from the full dataset
save_dir = '.'        # directory to save the new dataset files after preprocessing
K = 35                # value of K for the K-core graph


# Read in data files from Spotify dataset
data_files = os.listdir(data_dir)
data_files = sorted(data_files, key=lambda x: int(x.split(".")[2].split("-")[0]))
data_files = data_files[:NUM_FILES_TO_USE]

# Create undirected SNAP graph
G = snap.TUNGraph().New()
print("Works")
