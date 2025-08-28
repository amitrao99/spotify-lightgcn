
import json
import numpy as np
import os
import torch
from torch_geometric import seed_everything
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import degree
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from sklearn.decomposition import PCA
from IPython.display import HTML
import time  # Added for timing

# -------------------------------
# AMP (torch.amp) + misc perf flags
# -------------------------------
try:
    from torch.amp import autocast, GradScaler  # Use torch.amp (backend-agnostic; ROCm uses 'cuda')
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # Fallback for older PyTorch



seed_everything(5) # set random seed

import matplotlib

# -------------------------------
# Load data (unchanged logic)
# -------------------------------
base_dir = "."
data = torch.load(os.path.join(base_dir, "data_object.pt"),weights_only=False)
with open(os.path.join(base_dir, "dataset_stats.json"), 'r') as f:
    stats = json.load(f)
num_playlists, num_nodes = stats["num_playlists"], stats["num_nodes"]

# Train/val/test split (unchanged logic)
transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio=0,
                            num_val=0.15, num_test=0.15)
train_split, val_split, test_split = transform(data)
# Confirm that every node appears in every set above
assert train_split.num_nodes == val_split.num_nodes and train_split.num_nodes == test_split.num_nodes

print(train_split)
print(val_split)
print(test_split)

# -------------------------------
# Precompute GPU-resident normalization/edge weights for LightGCN
# -------------------------------
def build_mp(split: Data, device: torch.device) -> Data:
    """
    Precompute norms on GPU once and keep the graph resident to avoid repeated CPU<->GPU traffic
    and redundant degree/norm recomputation each forward pass.
    """
    edge_index = split.edge_index.contiguous().to(device)
    row, col = edge_index
    deg = degree(col, data.num_nodes, dtype=torch.float32)  # lives on same device as 'col'
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = (deg_inv_sqrt[row] * deg_inv_sqrt[col]).contiguous()
    return Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=data.num_nodes)

# -------------------------------
# Optional: popularity-based song distribution for GPU negative sampling
# (disabled by default to preserve behavior; can be enabled via NEG_SAMPLING_MODE="popularity")
# -------------------------------
def build_song_popularity_probs(edge_index: torch.Tensor, num_playlists: int, num_nodes: int, device: torch.device):
    """
    Builds a probability distribution over SONGS (indices [num_playlists, num_nodes)) based on their frequency
    in the training edges. Smoothing is applied to avoid zero-probability items.
    Returns a 1D tensor of length num_songs (sum to 1.0) on 'device'.
    """
    with torch.no_grad():
        # Use only edges where the source is a playlist to avoid counting both directions in undirected graphs.
        mask = edge_index[0, :] < num_playlists
        songs = edge_index[1, mask]  # absolute indices in [num_playlists, num_nodes)
        songs_rel = songs - num_playlists  # shift to [0, num_songs)
        num_songs = num_nodes - num_playlists
        counts = torch.bincount(songs_rel, minlength=num_songs).float()
        counts = counts + 1.0  # add-1 smoothing
        probs = counts / counts.sum()
        return probs.to(device)

class PlainData(Data):
    """
    Custom Data class for use in PyG. Basically the same as the original Data class from PyG, but
    overrides the __inc__ method because otherwise the DataLoader was incrementing indices unnecessarily.
    Now it functions more like the original DataLoader from PyTorch itself.
    See here for more information: https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    """
    def __inc__(self, key, value, *args, **kwargs):
        return 0

class SpotifyDataset(Dataset):
    """
    Dataset object containing the Spotify supervision/evaluation edges. This will be used by the DataLoader to load
    batches of edges to calculate loss or evaluation metrics on. Here, get(idx) will return ALL outgoing edges of the graph
    corresponding to playlist "idx." This is because when calculating metrics such as recall@k, we need all of the
    playlist's positive edges in the same batch.
    """
    def __init__(self, root, edge_index, transform=None, pre_transform=None):
        self.edge_index = edge_index
        self.unique_idxs = torch.unique(edge_index[0,:]).tolist() # playlists will all be in row 0, b/c sorted by RandLinkSplit
        self.num_nodes = len(self.unique_idxs)
        super().__init__(root, transform, pre_transform)

    def len(self):
        return self.num_nodes

    def get(self, idx): # returns all outgoing edges associated with playlist idx
        edge_index = self.edge_index[:, self.edge_index[0,:] == idx]
        return PlainData(edge_index=edge_index)

# -------------------------------
# Use GPU if available (unchanged)
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch.backends, "cudnn"):
    try:
        torch.backends.cudnn.benchmark = True  # perf hint on NVIDIA; harmless guard elsewhere
    except Exception:
        pass

# Build message-passing graphs ONCE and keep them on device
train_mp = build_mp(train_split, device)
val_mp   = build_mp(val_split, device)
test_mp  = build_mp(test_split, device)

# Supervision/evaluation edge datasets (unchanged semantics)
train_ev = SpotifyDataset('temp', edge_index=train_split.edge_label_index)
val_ev   = SpotifyDataset('temp', edge_index=val_split.edge_label_index)
test_ev  = SpotifyDataset('temp', edge_index=test_split.edge_label_index)

# Precompute song popularity probs for optional popularity-based GPU negatives
song_pop_probs = build_song_popularity_probs(train_split.edge_index, num_playlists, num_nodes, device)

# -------------------------------
# LightGCN and model (minimal changes; use precomputed edge_weight)
# -------------------------------
class LightGCN(MessagePassing):
    """
    A single LightGCN layer. Extends the MessagePassing class from PyTorch Geometric
    """
    def __init__(self):
        super(LightGCN, self).__init__(aggr='add') # aggregation function is 'add'

    def message(self, x_j, norm):
        """
        Same logic; use precomputed 'norm' (edge_weight) passed in from propagate().
        """
        return norm.view(-1, 1) * x_j

    def forward(self, x, edge_index, edge_weight):
        """
        IMPORTANT CHANGE: We no longer recompute degrees here.
        We expect precomputed edge_weight and simply pass it through to propagate().
        """
        return self.propagate(edge_index, x=x, norm=edge_weight)


class GNN(torch.nn.Module):
    """
    Overall graph neural network. Consists of learnable user/item (i.e., playlist/song) embeddings
    and LightGCN layers.
    """

    def __init__(self, embedding_dim, num_nodes, num_playlists, num_layers):
        super(GNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes  # total number of nodes (songs + playlists) in dataset
        self.num_playlists = num_playlists  # total number of playlists in dataset
        self.num_layers = num_layers

        # NOTE: Per instruction, NO feature-aware initializations. Keep original init.
        self.embeddings = torch.nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)
        torch.nn.init.normal_(self.embeddings.weight, std=0.1)

        self.layers = torch.nn.ModuleList()  # LightGCN layers
        for _ in range(self.num_layers):
            self.layers.append(LightGCN())

        self.sigmoid = torch.sigmoid

    def forward(self):
        raise NotImplementedError("forward() has not been implemented for the GNN class. Do not use")

    def gnn_propagation(self, data_mp: Data):
        """
        Performs the linear embedding propagation (using the LightGCN layers) and calculates final (multi-scale) embeddings.
        """
        x = self.embeddings.weight  # layer-0 embeddings

        x_at_each_layer = [x]  # stores embeddings from each layer. Start with layer-0 embeddings
        for i in range(self.num_layers):  # now performing the GNN propagation
            x = self.layers[i](x, data_mp.edge_index, data_mp.edge_weight)
            x_at_each_layer.append(x)
        final_embs = torch.stack(x_at_each_layer, dim=0).mean(dim=0)  # take average to calculate multi-scale embeddings
        return final_embs

    def predict_scores(self, edge_index, embs):
        """
        Calculates predicted scores for each playlist/song pair in the list of edges. Uses dot product of their embeddings.
        """
        scores = embs[edge_index[0, :], :] * embs[edge_index[1, :], :]  # taking dot product for each playlist/song pair
        scores = scores.sum(dim=1)
        scores = self.sigmoid(scores)
        return scores

    def calc_loss(self, data_mp, data_pos, data_neg):
        """
        Training step (unchanged semantics). Uses precomputed data_mp edge weights.
        """
        final_embs = self.gnn_propagation(data_mp)
        pos_scores = self.predict_scores(data_pos.edge_index, final_embs)
        neg_scores = self.predict_scores(data_neg.edge_index, final_embs)
        loss = -torch.log(self.sigmoid(pos_scores - neg_scores)).mean()
        return loss

    def evaluation(self, data_mp, data_pos, k):
        """
        Performs evaluation on validation or test set. Calculates recall@k.
        """
        final_embs = self.gnn_propagation(data_mp)

        # Get embeddings of all unique playlists in the batch of evaluation edges
        unique_playlists = torch.unique_consecutive(data_pos.edge_index[0, :])
        playlist_emb = final_embs[unique_playlists, :]  # has shape [number of playlists in batch, 64]

        # Get embeddings of ALL songs in dataset
        song_emb = final_embs[self.num_playlists:, :]  # has shape [total number of songs in dataset, 64]

        # All ratings for each playlist in batch to each song in entire dataset (using dot product as the scoring function)
        ratings = self.sigmoid(
            torch.matmul(playlist_emb, song_emb.t()))  # shape: [# playlists in batch, # songs in dataset]
        # where entry i,j is rating of song j for playlist i
        # Calculate recall@k
        result = recall_at_k(ratings.cpu(), k, self.num_playlists, data_pos.edge_index.cpu(),
                             unique_playlists.cpu(), data_mp.edge_index.cpu())
        return result


def recall_at_k(all_ratings, k, num_playlists, ground_truth, unique_playlists, data_mp):
    """
    Calculates recall@k during validation/testing for a single batch.
    (Unchanged logic)
    """
    # We don't want to recommend songs that are already known to be in the playlist!
    # Set those to a low rating so they won't be recommended
    known_edges = data_mp[:, data_mp[0,
                             :] < num_playlists]  # removing duplicate edges (since data_mp is undirected). also makes it so
    # that for each column, playlist idx is in row 0 and song idx is in row 1
    playlist_to_idx_in_batch = {playlist: i for i, playlist in enumerate(unique_playlists.tolist())}
    exclude_playlists, exclude_songs = [], []  # already-known playlist/song links. Don't want to recommend these again
    for i in range(known_edges.shape[1]):  # looping over all known edges
        pl, song = known_edges[:, i].tolist()
        if pl in playlist_to_idx_in_batch:  # don't need the edges in data_mp that are from playlists that are not in this batch
            exclude_playlists.append(playlist_to_idx_in_batch[pl])
            exclude_songs.append(
                song - num_playlists)  # subtract num_playlists to get indexing into all_ratings correct
    all_ratings[exclude_playlists, exclude_songs] = -10000  # setting to a very low score so they won't be recommended

    # Get top k recommendations for each playlist
    _, top_k = torch.topk(all_ratings, k=k, dim=1)
    top_k += num_playlists  # topk returned indices of songs in ratings, which doesn't include playlists.
    # Need to shift up by num_playlists to get the actual song indices

    # Calculate recall@k
    ret = {}
    for i, playlist in enumerate(unique_playlists):
        pos_songs = ground_truth[1, ground_truth[0, :] == playlist]

        k_recs = top_k[i, :]  # top k recommendations for playlist
        recall = len(np.intersect1d(pos_songs, k_recs)) / len(pos_songs)
        ret[playlist] = recall
    return ret


# -------------------------------
# GPU-based negative edge sampling (vectorized) with optional popularity-based mode
# -------------------------------
def sample_negative_edges_gpu(batch: Data,
                              num_playlists: int,
                              num_nodes: int,
                              device: torch.device,
                              negative_mode: str = "uniform",
                              song_pop_probs: torch.Tensor = None) -> Data:
    """
    Vectorized negative sampling directly on GPU.
    Modes:
      - "uniform"   : sample songs uniformly at random (matches baseline behavior, fastest)
      - "popularity": sample songs by popularity distribution (requires song_pop_probs over songs)
    Returns a Data(edge_index=...) object whose tensors live on the sampling device.
    """
    # Use the batch's actual device if already on GPU; otherwise fall back to the provided device
    playlists = batch.edge_index[0, :]
    sample_device = playlists.device if playlists.is_cuda else device

    # Popularity-weighted (optional)
    if negative_mode == "popularity" and song_pop_probs is not None:
        probs = song_pop_probs.to(sample_device, non_blocking=True)
        rel_idx = torch.multinomial(probs, num_samples=playlists.numel(), replacement=True)
        neg_songs = rel_idx + num_playlists
    else:
        # Uniform GPU sampling
        neg_songs = torch.randint(low=num_playlists, high=num_nodes,
                                  size=(playlists.numel(),), device=sample_device)

    # Ensure playlists index is on the same device as negatives
    if playlists.device != sample_device:
        playlists = playlists.to(sample_device, non_blocking=True)

    edge_index_negs = torch.stack((playlists, neg_songs), dim=0).contiguous()
    return Data(edge_index=edge_index_negs)


# -------------------------------
# Train / Test with AMP, DataLoader perf knobs, torch.compile
# -------------------------------
def train(model, data_mp, loader, opt, num_playlists, num_nodes, device, batch_size,
          use_amp=True, amp_dtype=torch.float16, scaler=None, negative_mode="uniform", song_pop_probs=None):
    """
    Main training loop with GPU-based negative sampling.
    """
    total_loss = 0
    total_examples = 0
    model.train()
    epoch_start_time = time.time()  # Start timing the epoch
    
    for batch in loader:
        del batch.batch
        del batch.ptr  # delete unwanted attributes

        opt.zero_grad(set_to_none=True)

        # Move minibatch to GPU first; then sample negatives on GPU (vectorized)
        batch = batch.to(device)
        negs = sample_negative_edges_gpu(batch, num_playlists, num_nodes, device,
                                         negative_mode=negative_mode, song_pop_probs=song_pop_probs)

        data_mp = data_mp.to(device)  # no-op if already on device

        if use_amp and scaler is not None:
            with autocast(device_type="cuda", dtype=amp_dtype):
                loss = model.calc_loss(data_mp, batch, negs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = model.calc_loss(data_mp, batch, negs)
            loss.backward()
            opt.step()

        num_examples = batch.edge_index.shape[1]
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    epoch_time = time.time() - epoch_start_time  # Calculate epoch time
    avg_loss = total_loss / total_examples
    return avg_loss, epoch_time, total_examples


def test(model, data_mp, loader, k, device, save_dir, epoch, batch_size,
         use_amp=True, amp_dtype=torch.float16):
    """
    Evaluation loop for validation/testing.
    (Semantics/logging preserved; added autocast for eval)
    """
    model.eval()
    all_recalls = {}
    eval_start_time = time.time()  # Start timing evaluation
    total_edges = 0
    
    with torch.no_grad():
        # Save multi-scale embeddings if save_dir is not None
        data_mp = data_mp.to(device)
        if save_dir is not None:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                embs_to_save = gnn.gnn_propagation(data_mp)
            torch.save(embs_to_save, os.path.join(save_dir, f"embeddings_epoch_{epoch}.pt"))

        # Run evaluation
        for batch in loader:
            del batch.batch; del batch.ptr # delete unwanted attributes
            batch = batch.to(device)
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                recalls = model.evaluation(data_mp, batch, k)
            total_edges += batch.edge_index.shape[1]  # Count edges in this batch
            for playlist_idx in recalls:
                assert playlist_idx not in all_recalls
            all_recalls.update(recalls)
    
    eval_time = time.time() - eval_start_time  # Calculate evaluation time
    recall_at_k_val = np.mean(list(all_recalls.values()))
    return recall_at_k_val, eval_time, total_edges


num_songs = num_nodes - num_playlists
print(f"There are {num_songs} unique songs in the dataset")
print (300 / num_songs)

# -------------------------------
# Training hyperparameters (unchanged)
# -------------------------------
epochs = 30        # number of training epochs
k = 300            # value of k for recall@k
num_layers = 3     # number of LightGCN layers
batch_size = 512  # batch size (playlists per batch)
embedding_dim = 64 # dimension for embeddings
save_emb_dir = 'embeddings'  # path to save multi-scale embeddings during test(). If None, will not save any embeddings

# Negative sampling mode: "uniform" (default, fastest) or "popularity" (optional, quality-oriented)
NEG_SAMPLING_MODE = "uniform"  # change to "popularity" to enable popularity-weighted GPU sampling

# AMP settings
use_amp = torch.cuda.is_available()
amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
scaler = GradScaler(enabled=use_amp)

# Print all hyperparameters (unchanged block)
print("\n" + "="*60)
print("HYPERPARAMETERS")
print("="*60)
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: 1e-3")
print(f"Optimizer: Adam")
print(f"Embedding dimension: {embedding_dim}")
print(f"Number of LightGCN layers: {num_layers}")
print(f"k for recall@k: {k}")
print(f"Device: {device}")
print(f"Loss function: BPR (Bayesian Personalized Ranking)")
print(f"Negative sampling ratio: 1:1")
print(f"Negative sampling mode: {NEG_SAMPLING_MODE}")
print(f"Embeddings save directory: {save_emb_dir}")
print("="*60)

# Make directory to save embeddings (unchanged behavior)
if save_emb_dir is not None:
  os.makedirs(save_emb_dir, exist_ok=True)

# -------------------------------
# DataLoader perf settings (num_workers/pin/prefetch/persistent)
# -------------------------------
loader_kw = dict(pin_memory=True, num_workers=6, prefetch_factor=32, persistent_workers=True)

train_loader = DataLoader(train_ev, batch_size=batch_size, shuffle=True, **loader_kw)
val_loader   = DataLoader(val_ev,   batch_size=2048*4, shuffle=False, **loader_kw)
test_loader  = DataLoader(test_ev,  batch_size=2048*4, shuffle=False, **loader_kw)

# Initialize GNN model (unchanged semantics)
gnn = GNN(embedding_dim=embedding_dim, num_nodes=data.num_nodes, num_playlists=num_playlists, num_layers=num_layers).to(device)

# -------------------------------
# torch.compile with better mode
# -------------------------------
try:
    gnn = torch.compile(gnn, mode='reduce-overhead', fullgraph=True)
except Exception as e:
    print(f"torch.compile not available or failed ({e}); continuing without compile.")

# Count and print model parameters (unchanged)
total_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")
print("="*60 + "\n")

opt = torch.optim.Adam(gnn.parameters(), lr=1e-3) # using Adam optimizer

all_train_losses = []  # list of (epoch, training loss)
all_val_recalls = []  # list of (epoch, validation recall@k)

# Main training loop
for epoch in range(epochs):
    train_loss, train_time, train_edges = train(gnn, train_mp, train_loader, opt, num_playlists, num_nodes, device, batch_size,
                                                use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler,
                                                negative_mode=NEG_SAMPLING_MODE, song_pop_probs=song_pop_probs)
    all_train_losses.append((epoch, train_loss))
    
    # Calculate training metrics
    num_batches = len(train_loader)
    avg_batch_time = train_time / num_batches if num_batches > 0 else float('inf')
    edges_per_sec = train_edges / train_time if train_time > 0 else 0.0

    if epoch in range(0) or epoch % 10 == 0:  # perform validation for the first ~10 epochs, then every 5 epochs after that
        val_recall, val_time, val_edges = test(gnn, val_mp, val_loader, k, device, save_emb_dir, epoch, batch_size,
                                               use_amp=use_amp, amp_dtype=amp_dtype)
        all_val_recalls.append((epoch, val_recall))
        
        # Calculate validation metrics
        val_num_batches = len(val_loader)
        val_avg_batch_time = val_time / val_num_batches if val_num_batches > 0 else float('inf')
        val_edges_per_sec = val_edges / val_time if val_time > 0 else 0.0
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_recall={val_recall:.4f}")
        print(f"  Training   - Batch size: {batch_size}, Time: {train_time:.2f}s, Avg batch time: {avg_batch_time:.3f}s, Throughput: {edges_per_sec:.0f} edges/sec")
        print(f"  Validation - Batch size: {batch_size}, Time: {val_time:.2f}s, Avg batch time: {val_avg_batch_time:.3f}s, Throughput: {val_edges_per_sec:.0f} edges/sec")
    else:
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        print(f"  Training   - Batch size: {batch_size}, Time: {train_time:.2f}s, Avg batch time: {avg_batch_time:.3f}s, Throughput: {edges_per_sec:.0f} edges/sec")

print()

# Print best validation recall@k value (unchanged)
best_val_recall = max(all_val_recalls, key=lambda x: x[1])
print(f"Best validation recall@k: {best_val_recall[1]} at epoch {best_val_recall[0]}")

# Print final recall@k on test set with timing (unchanged)
print("\nEvaluating on test set...")
test_recall, test_time, test_edges = test(gnn, test_mp, test_loader, k, device, None, None, batch_size,
                                          use_amp=use_amp, amp_dtype=amp_dtype)
test_num_batches = len(test_loader)
test_avg_batch_time = test_time / test_num_batches if test_num_batches > 0 else float('inf')
test_edges_per_sec = test_edges / test_time if test_time > 0 else 0.0

print(f"Test set recall@k: {test_recall:.4f}")
print(f"Test evaluation - Batch size: {batch_size}, Time: {test_time:.2f}s, Avg batch time: {test_avg_batch_time:.3f}s, Throughput: {test_edges_per_sec:.0f} edges/sec")

# Plot (unchanged)
plt.plot([x[0] for x in all_train_losses], [x[1] for x in all_train_losses])
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.show()
