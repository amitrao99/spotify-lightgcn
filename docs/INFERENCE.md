Inference (Top‑K Recommendations)
================================

This guide shows how to produce top‑k song recommendations for a given playlist using the LightGCN model logic present in each script.


Approach
--------

The scripts already define:

- `GNN.gnn_propagation(edge_index)` — returns multi‑scale embeddings for all nodes.
- `GNN.predict_scores(edge_index, embs)` — scores playlist–song pairs via dot product.
- `evaluation(...)` — runs recall@k (includes logic to exclude already‑known edges from recommendations).

To get recommendations for one playlist ID, we can adapt the evaluation logic as follows:

1) Load the graph and dataset stats.
2) Recreate the model with the same hyperparameters used in training (embedding dim, layers, etc.).
3) Load weights if you saved a checkpoint (some variants include checkpoints in their folders).
4) Run propagation to get node embeddings.
5) Score all songs for the target playlist and take the top‑k, excluding already‑known songs for that playlist.


Minimal Example (Base Setup)
----------------------------

Run this from the repository root after training (adjust paths, hyperparameters, and playlist_id as needed):

```python
import os, json, torch
from BaseSetup.base_model_lightgcn import GNN

# 1) Load graph and stats
data = torch.load('BaseSetup/data_object.pt', weights_only=False)
with open('BaseSetup/dataset_stats.json') as f:
    stats = json.load(f)
num_playlists, num_nodes = stats['num_playlists'], stats['num_nodes']

# 2) Recreate the model (match training hyperparameters)
embedding_dim = 64
num_layers = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = GNN(embedding_dim=embedding_dim, num_nodes=num_nodes, num_playlists=num_playlists, num_layers=num_layers).to(device)

# 3) (Optional) Load saved weights if available
# gnn.load_state_dict(torch.load('path/to/checkpoint.pt', map_location=device))
gnn.eval()

# 4) Compute multi‑scale embeddings
edge_index_mp = data.edge_index.to(device)
embs = gnn.gnn_propagation(edge_index_mp)

# 5) Recommend top‑k for a given playlist
playlist_id = 0            # choose a playlist index in [0, num_playlists)
k = 20                     # how many songs to recommend
song_emb = embs[num_playlists:, :]                 # all songs
pl_emb = embs[playlist_id:playlist_id+1, :]        # single playlist
scores = torch.matmul(pl_emb, song_emb.T).squeeze()  # dot product scores

# Exclude songs already linked to this playlist (don’t re‑recommend known items)
known = data.edge_index[:, data.edge_index[0, :] == playlist_id]
known_song_ids = known[1, :] - num_playlists
scores[known_song_ids] = -1e9

topk_vals, topk_idx = torch.topk(scores, k=k)
recommended_song_node_ids = (topk_idx + num_playlists).tolist()
print('Top‑k song node IDs:', recommended_song_node_ids)
```

Notes
-----

- The example uses the Base Setup class for convenience. For other variants, import the corresponding model class or reuse the same method calls inside that script.
- If you trained on GPU, ensure consistent device placement when loading weights and tensors.
- To map node IDs to track metadata, use `song_info.json` (e.g., artist/title) from the same folder.

