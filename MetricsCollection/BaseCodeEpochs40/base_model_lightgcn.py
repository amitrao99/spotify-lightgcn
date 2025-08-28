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
# Import the metrics tracker
import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
best_val = float("-inf")
best_epoch = -1
best_ckpt = "best_baseline.pt"
stop_on_first_best = False  # set True if you want to immediately stop at the first improvement

class MetricsTracker:
    """
    Universal metrics tracker that works across all model variants.
    Records training metrics in a consistent format for comparative analysis.
    """
    
    def __init__(self, model_name, save_dir="experiment_results"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize tracking dictionaries
        self.metrics = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": {},
            "model_stats": {},
            "training_history": {
                "epoch": [],
                "train_loss": [],
                "val_recall": [],
                "epoch_time": [],
                "edges_per_sec": [],
                "gpu_memory_mb": []
            },
            "test_results": {},
            "per_playlist_recalls": {},
            "computational_stats": {
                "total_training_time": 0,
                "avg_epoch_time": 0,
                "peak_gpu_memory_gb": 0
            }
        }
        
        self.start_time = None
        self.epoch_start = None
        
    def log_hyperparameters(self, **kwargs):
        """Log hyperparameters at the start of training"""
        self.metrics["hyperparameters"] = kwargs
        
    def log_model_stats(self, model):
        """Log model architecture statistics"""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.metrics["model_stats"] = {
            "total_trainable_params": total_params,
            "total_params": sum(p.numel() for p in model.parameters()),
            "embedding_params": 0,  # Will be updated if available
            "layer_params": 0  # Will be updated if available
        }
        
        # Try to get embedding params if model has embeddings attribute
        if hasattr(model, 'embeddings') or hasattr(model, 'emb'):
            emb_attr = getattr(model, 'embeddings', None) or getattr(model, 'emb', None)
            if emb_attr:
                self.metrics["model_stats"]["embedding_params"] = sum(
                    p.numel() for p in emb_attr.parameters()
                )
        
        return total_params
        
    def start_training(self):
        """Mark the start of training"""
        self.start_time = time.time()
        
    def start_epoch(self):
        """Mark the start of an epoch"""
        self.epoch_start = time.time()
        
    def end_epoch(self, epoch, train_loss, val_recall=None, train_edges=None, train_time=None):
        """
        Log metrics at the end of an epoch
        """
        # Use provided train_time if available, otherwise calculate from epoch_start
        if train_time is not None:
            epoch_time = train_time
        else:
            epoch_time = time.time() - self.epoch_start if self.epoch_start else 0
        
        # Record basic metrics
        self.metrics["training_history"]["epoch"].append(epoch)
        self.metrics["training_history"]["train_loss"].append(float(train_loss))
        self.metrics["training_history"]["epoch_time"].append(epoch_time)
        
        # Record validation recall (use -1 if not evaluated this epoch)
        if val_recall is not None:
            self.metrics["training_history"]["val_recall"].append(float(val_recall))
        else:
            # Use previous value or -1 if first epoch without validation
            prev_val = self.metrics["training_history"]["val_recall"][-1] if self.metrics["training_history"]["val_recall"] else -1
            self.metrics["training_history"]["val_recall"].append(prev_val)
        
        # Calculate edges per second
        if train_edges and epoch_time > 0:
            edges_per_sec = train_edges / epoch_time
            self.metrics["training_history"]["edges_per_sec"].append(edges_per_sec)
        else:
            self.metrics["training_history"]["edges_per_sec"].append(0)
        
        # Try to get GPU memory usage
        if torch.cuda.is_available():
            try:
                # Get current GPU memory in MB
                gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                self.metrics["training_history"]["gpu_memory_mb"].append(gpu_mem_mb)
                
                # Update peak memory
                peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
                self.metrics["computational_stats"]["peak_gpu_memory_gb"] = max(
                    self.metrics["computational_stats"]["peak_gpu_memory_gb"],
                    peak_gb
                )
            except:
                self.metrics["training_history"]["gpu_memory_mb"].append(0)
        else:
            self.metrics["training_history"]["gpu_memory_mb"].append(0)
            
    def end_training(self):
        """Mark the end of training and calculate summary stats"""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.metrics["computational_stats"]["total_training_time"] = total_time
            self.metrics["computational_stats"]["total_training_hours"] = total_time / 3600
            
            # Calculate average epoch time
            epoch_times = self.metrics["training_history"]["epoch_time"]
            if epoch_times:
                self.metrics["computational_stats"]["avg_epoch_time"] = np.mean(epoch_times)
                
    def log_test_results(self, test_recall, test_time=None):
        """Log final test results"""
        self.metrics["test_results"] = {
            "final_test_recall": float(test_recall),
            "test_evaluation_time": test_time if test_time else 0,
            "best_val_recall": max(self.metrics["training_history"]["val_recall"]) if self.metrics["training_history"]["val_recall"] else 0,
            "best_val_epoch": int(np.argmax(self.metrics["training_history"]["val_recall"])) if self.metrics["training_history"]["val_recall"] else -1
        }
        
    def log_per_playlist_recalls(self, recall_dict):
        """Log per-playlist recall values for distribution analysis"""
        # Convert to list of floats for JSON serialization
        recalls_list = [float(v) for v in recall_dict.values()]
        self.metrics["per_playlist_recalls"] = {
            "values": recalls_list,
            "mean": float(np.mean(recalls_list)),
            "std": float(np.std(recalls_list)),
            "min": float(np.min(recalls_list)),
            "max": float(np.max(recalls_list)),
            "median": float(np.median(recalls_list)),
            "q25": float(np.percentile(recalls_list, 25)),
            "q75": float(np.percentile(recalls_list, 75))
        }
        
    def save(self):
        """Save all metrics to a JSON file"""
        filename = self.save_dir / f"{self.model_name}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\nMetrics saved to: {filename}")
        return filename
        
    def get_summary_table_row(self):
        """Get a summary row for the comparison table"""
        return {
            "Model": self.model_name,
            "Trainable Parameters": self.metrics["model_stats"]["total_trainable_params"],
            "Avg Epoch Time (s)": round(self.metrics["computational_stats"]["avg_epoch_time"], 2),
            "Total Training Time (h)": round(self.metrics["computational_stats"]["total_training_hours"], 2),
            "Peak GPU Memory (GB)": round(self.metrics["computational_stats"]["peak_gpu_memory_gb"], 2),
            "Best Val Recall@300": round(self.metrics["test_results"]["best_val_recall"], 4),
            "Test Recall@300": round(self.metrics["test_results"]["final_test_recall"], 4),
            "Best Epoch": self.metrics["test_results"]["best_val_epoch"]
        }
seed_everything(5) # set random seed


import matplotlib



# Load data
base_dir = "."
data = torch.load(os.path.join(base_dir, "data_object.pt"),weights_only=False)
with open(os.path.join(base_dir, "dataset_stats.json"), 'r') as f:
    stats = json.load(f)
num_playlists, num_nodes = stats["num_playlists"], stats["num_nodes"]



# Train/val/test split
transform = RandomLinkSplit(is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio=0,
                            num_val=0.15, num_test=0.15)
train_split, val_split, test_split = transform(data)
# Confirm that every node appears in every set above
assert train_split.num_nodes == val_split.num_nodes and train_split.num_nodes == test_split.num_nodes

print(train_split)
print(val_split)
print(test_split)



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




train_ev = SpotifyDataset('temp', edge_index=train_split.edge_label_index)
train_mp = Data(edge_index=train_split.edge_index)

val_ev = SpotifyDataset('temp', edge_index=val_split.edge_label_index)
val_mp = Data(edge_index=val_split.edge_index)

test_ev = SpotifyDataset('temp', edge_index=test_split.edge_label_index)
test_mp = Data(edge_index=test_split.edge_index)




class LightGCN(MessagePassing):
    """
    A single LightGCN layer. Extends the MessagePassing class from PyTorch Geometric
    """
    def __init__(self):
        super(LightGCN, self).__init__(aggr='add') # aggregation function is 'add

    def message(self, x_j, norm):
        """
        Specifies how to perform message passing during GNN propagation. For LightGCN, we simply pass along each
        source node's embedding to the target node, normalized by the normalization term for that node.
        args:
          x_j: node embeddings of the neighbor nodes, which will be passed to the central node (shape: [E, emb_dim])
          norm: the normalization terms we calculated in forward() and passed into propagate()
        returns:
          messages from neighboring nodes j to central node i
        """
        # Here we are just multiplying the x_j's by the normalization terms (using some broadcasting)
        return norm.view(-1, 1) * x_j

    def forward(self, x, edge_index):
        """
        Performs the LightGCN message passing/aggregation/update to get updated node embeddings

        args:
          x: current node embeddings (shape: [N, emb_dim])
          edge_index: message passing edges (shape: [2, E])
        returns:
          updated embeddings after this layer
        """
        # Computing node degrees for normalization term in LightGCN (see LightGCN paper for details on this normalization term)
        # These will be used during message passing, to normalize each neighbor's embedding before passing it as a message
        row, col = edge_index
        deg = degree(col)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Begin propagation. Will perform message passing and aggregation and return updated node embeddings.
        return self.propagate(edge_index, x=x, norm=norm)


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

        # Initialize embeddings for all playlists and songs. Playlists will have indices from 0...num_playlists-1,
        # songs will have indices from num_playlists...num_nodes-1
        self.embeddings = torch.nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)
        torch.nn.init.normal_(self.embeddings.weight, std=0.1)

        self.layers = torch.nn.ModuleList()  # LightGCN layers
        for _ in range(self.num_layers):
            self.layers.append(LightGCN())

        self.sigmoid = torch.sigmoid

    def forward(self):
        raise NotImplementedError("forward() has not been implemented for the GNN class. Do not use")

    def gnn_propagation(self, edge_index_mp):
        """
        Performs the linear embedding propagation (using the LightGCN layers) and calculates final (multi-scale) embeddings
        for each user/item, which are calculated as a weighted sum of that user/item's embeddings at each layer (from
        0 to self.num_layers). Technically, the weighted sum here is the average, which is what the LightGCN authors recommend.

        args:
          edge_index_mp: a tensor of all (undirected) edges in the graph, which is used for message passing/propagation and
              calculating the multi-scale embeddings. (In contrast to the evaluation/supervision edges, which are distinct
              from the message passing edges and will be used for calculating loss/performance metrics).
        returns:
          final multi-scale embeddings for all users/items
        """
        x = self.embeddings.weight  # layer-0 embeddings

        x_at_each_layer = [x]  # stores embeddings from each layer. Start with layer-0 embeddings
        for i in range(self.num_layers):  # now performing the GNN propagation
            x = self.layers[i](x, edge_index_mp)
            x_at_each_layer.append(x)
        final_embs = torch.stack(x_at_each_layer, dim=0).mean(dim=0)  # take average to calculate multi-scale embeddings
        return final_embs

    def predict_scores(self, edge_index, embs):
        """
        Calculates predicted scores for each playlist/song pair in the list of edges. Uses dot product of their embeddings.

        args:
          edge_index: tensor of edges (between playlists and songs) whose scores we will calculate.
          embs: node embeddings for calculating predicted scores (typically the multi-scale embeddings from gnn_propagation())
        returns:
          predicted scores for each playlist/song pair in edge_index
        """
        scores = embs[edge_index[0, :], :] * embs[edge_index[1, :], :]  # taking dot product for each playlist/song pair
        scores = scores.sum(dim=1)
        scores = self.sigmoid(scores)
        return scores

    def calc_loss(self, data_mp, data_pos, data_neg):
        """
        The main training step. Performs GNN propagation on message passing edges, to get multi-scale embeddings.
        Then predicts scores for each training example, and calculates Bayesian Personalized Ranking (BPR) loss.

        args:
          data_mp: tensor of edges used for message passing / calculating multi-scale embeddings
          data_pos: set of positive edges that will be used during loss calculation
          data_neg: set of negative edges that will be used during loss calculation
        returns:
          loss calculated on the positive/negative training edges
        """
        # Perform GNN propagation on message passing edges to get final embeddings
        final_embs = self.gnn_propagation(data_mp.edge_index)

        # Get edge prediction scores for all positive and negative evaluation edges
        pos_scores = self.predict_scores(data_pos.edge_index, final_embs)
        neg_scores = self.predict_scores(data_neg.edge_index, final_embs)

        # # Calculate loss (binary cross-entropy). Commenting out, but can use instead of BPR if desired.
        # all_scores = torch.cat([pos_scores, neg_scores], dim=0)
        # all_labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])], dim=0)
        # loss_fn = torch.nn.BCELoss()
        # loss = loss_fn(all_scores, all_labels)

        # Calculate loss (using variation of Bayesian Personalized Ranking loss, similar to the one used in official
        # LightGCN implementation at https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py#L202)
        loss = -torch.log(self.sigmoid(pos_scores - neg_scores)).mean()
        return loss

    def evaluation(self, data_mp, data_pos, k):
        """
        Performs evaluation on validation or test set. Calculates recall@k.

        args:
          data_mp: message passing edges to use for propagation/calculating multi-scale embeddings
          data_pos: positive edges to use for scoring metrics. Should be no overlap between these edges and data_mp's edges
          k: value of k to use for recall@k
        returns:
          dictionary mapping playlist ID -> recall@k on that playlist
        """
        # Run propagation on the message-passing edges to get multi-scale embeddings
        final_embs = self.gnn_propagation(data_mp.edge_index)

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

    args:
      all_ratings: array of shape [number of playlists in batch, number of songs in whole dataset]
      k: the value of k to use for recall@k
      num_playlists: the number of playlists in the dataset
      ground_truth: array of shape [2, X] where each column is a pair of (playlist_idx, positive song idx). This is the
         batch that we are calculating metrics on.
      unique_playlists: 1D vector of length [number of playlists in batch], which specifies which playlist corresponds
         to each row of all_ratings
      data_mp: an array of shape [2, Y]. This is all of the known message-passing edges. We will use this to make sure we
         don't recommend songs that are already known to be in the playlist.
    returns:
      Dictionary of playlist ID -> recall@k on that playlist
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


def sample_negative_edges(batch, num_playlists, num_nodes):
    # Randomly samples songs for each playlist. Here we sample 1 negative edge
    # for each positive edge in the graph, so we will
    # end up having a balanced 1:1 ratio of positive to negative edges.
    negs = []
    for i in batch.edge_index[0,:]:  # looping over playlists
        assert i < num_playlists     # just ensuring that i is a playlist
        rand = torch.randint(num_playlists, num_nodes, (1,))  # randomly sample a song
        negs.append(rand.item())
    edge_index_negs = torch.row_stack([batch.edge_index[0,:], torch.LongTensor(negs)])
    return Data(edge_index=edge_index_negs)


def train(model, data_mp, loader, opt, num_playlists, num_nodes, device, batch_size):
    """
    Main training loop

    args:
       model: the GNN model
       data_mp: message passing edges to use for performing propagation/calculating multi-scale embeddings
       loader: DataLoader that loads in batches of supervision/evaluation edges
       opt: the optimizer
       num_playlists: the number of playlists in the entire dataset
       num_nodes: the number of nodes (playlists + songs) in the entire dataset
       device: whether to run on CPU or GPU
       batch_size: batch size for logging
    returns:
       the training loss for this epoch, total time, total edges processed
    """
    total_loss = 0
    total_examples = 0
    model.train()
    epoch_start_time = time.time()  # Start timing the epoch
    
    for batch in loader:
        del batch.batch;
        del batch.ptr  # delete unwanted attributes

        opt.zero_grad()
        negs = sample_negative_edges(batch, num_playlists, num_nodes)  # sample negative edges
        data_mp, batch, negs = data_mp.to(device), batch.to(device), negs.to(device)
        loss = model.calc_loss(data_mp, batch, negs)
        loss.backward()
        opt.step()

        num_examples = batch.edge_index.shape[1]
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    epoch_time = time.time() - epoch_start_time  # Calculate epoch time
    avg_loss = total_loss / total_examples
    return avg_loss, epoch_time, total_examples


def test(model, data_mp, loader, k, device, save_dir, epoch, batch_size):
    model.eval()
    all_recalls = {}
    eval_start_time = time.time()
    total_edges = 0

    with torch.no_grad():
        data_mp = data_mp.to(device)
        if save_dir is not None:
            embs_to_save = model.gnn_propagation(data_mp.edge_index)  # <- fixed
            torch.save(embs_to_save, os.path.join(save_dir, f"embeddings_epoch_{epoch}.pt"))

        for batch in loader:
            del batch.batch; del batch.ptr
            batch = batch.to(device)
            recalls = model.evaluation(data_mp, batch, k)
            total_edges += batch.edge_index.shape[1]
            for playlist_idx in recalls:
                assert playlist_idx not in all_recalls
            all_recalls.update(recalls)

    eval_time = time.time() - eval_start_time
    recall_at_k = np.mean(list(all_recalls.values())) if all_recalls else 0.0
    return recall_at_k, eval_time, total_edges
num_songs = num_nodes - num_playlists
print(f"There are {num_songs} unique songs in the dataset")
print (300 / num_songs)



# Training hyperparameters
epochs = 40        # number of training epochs (we are keeping it relatively low so that this Colab runs fast)
k = 300            # value of k for recall@k. It is important to set this to a reasonable value!
num_layers = 3     # number of LightGCN layers (i.e., number of hops to consider during propagation)
batch_size = 2048  # batch size. refers to the # of playlists in the batch (each will come with all of its edges)
embedding_dim = 64 # dimension to use for the playlist/song embeddings
save_emb_dir = 'embeddings'  # path to save multi-scale embeddings during test(). If None, will not save any embeddings


tracker = MetricsTracker("baseline")
tracker.log_hyperparameters(
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=1e-3,
    embedding_dim=embedding_dim,
    num_layers=num_layers,
    k=k,
    optimizer="Adam",
    loss_function="BPR",
    negative_sampling_ratio="1:1"
)


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print all hyperparameters
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
print(f"Embeddings save directory: {save_emb_dir}")
print("="*60)

# Make directory to save embeddings
if save_emb_dir is not None:
  os.mkdir(save_emb_dir)


train_loader = DataLoader(train_ev, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ev, batch_size=batch_size*32, shuffle=False)
test_loader = DataLoader(test_ev, batch_size=batch_size*32, shuffle=False)
# Initialize GNN model
gnn = GNN(embedding_dim=embedding_dim, num_nodes=data.num_nodes, num_playlists=num_playlists, num_layers=num_layers).to(device)
tracker = MetricsTracker("baseline")
tracker.log_hyperparameters(
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=1e-3,
    embedding_dim=embedding_dim,
    num_layers=num_layers,
    k=k,
    optimizer="Adam",
    loss_function="BPR",
    negative_sampling_ratio="1:1"
)
tracker.log_model_stats(gnn)
tracker.start_training()


# Count and print model parameters
total_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")
print("="*60 + "\n")

opt = torch.optim.Adam(gnn.parameters(), lr=1e-3) # using Adam optimizer

all_train_losses = []  # list of (epoch, training loss)
all_val_recalls = []  # list of (epoch, validation recall@k)
for epoch in range(epochs):
    tracker.start_epoch()  # ADD THIS
    
    train_loss, train_time, train_edges = train(gnn, train_mp, train_loader, opt, num_playlists, num_nodes, device, batch_size)
    all_train_losses.append((epoch, train_loss))
    
    # Calculate training metrics
    num_batches = len(train_loader)
    avg_batch_time = train_time / num_batches
    edges_per_sec = train_edges / train_time

    if epoch in range(11) or epoch % 5 == 0:  # perform validation for the first ~10 epochs, then every 5 epochs after that
        val_recall, val_time, val_edges = test(gnn, val_mp, val_loader, k, device, save_emb_dir, epoch, batch_size)
        all_val_recalls.append((epoch, val_recall))
        
        tracker.end_epoch(epoch, train_loss, val_recall, train_edges, train_time)  # ADD THIS - Note: passing train_time
        
        # Calculate validation metrics
        val_num_batches = len(val_loader)
        val_avg_batch_time = val_time / val_num_batches
        val_edges_per_sec = val_edges / val_time
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_recall={val_recall:.4f}")
        print(f"  Training   - Batch size: {batch_size}, Time: {train_time:.2f}s, Avg batch time: {avg_batch_time:.3f}s, Throughput: {edges_per_sec:.0f} edges/sec")
        print(f"  Validation - Batch size: {batch_size}, Time: {val_time:.2f}s, Avg batch time: {val_avg_batch_time:.3f}s, Throughput: {val_edges_per_sec:.0f} edges/sec")
        if val_recall > best_val:
            best_val = val_recall
            best_epoch = epoch
            torch.save({
                "model_state_dict": gnn.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "val_recall": float(val_recall),
            }, best_ckpt)
            print(f"  ↳ New BEST val_recall={val_recall:.4f} at epoch {epoch}. Saved {best_ckpt}")
            if stop_on_first_best:
                print("  ↳ Stopping on first best as requested.")
                break
    else:
        tracker.end_epoch(epoch, train_loss, train_edges=train_edges, train_time=train_time)  # ADD THIS - Note: passing train_time
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        print(f"  Training   - Batch size: {batch_size}, Time: {train_time:.2f}s, Avg batch time: {avg_batch_time:.3f}s, Throughput: {edges_per_sec:.0f} edges/sec")

tracker.end_training()  # ADD THIS AFTER THE LOOP
# Load best checkpoint for testing (matches first file’s behavior)
if os.path.exists(best_ckpt):
    ckpt = torch.load(best_ckpt, map_location=device)
    gnn.load_state_dict(ckpt["model_state_dict"])
    try:
        opt.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception:
        pass
    print(f"\nLoaded best checkpoint from epoch {ckpt.get('epoch','?')} "
          f"(val_recall={ckpt.get('val_recall','?')}). Testing with these weights.")
else:
    print("\n[Warning] No best checkpoint found; testing with last-epoch weights.")

# Optional: print best val from history (if you recorded it)
if all_val_recalls:
    best_val_epoch, best_val_value = max(all_val_recalls, key=lambda x: x[1])
    print(f"Best validation recall@{k}: {best_val_value:.4f} at epoch {best_val_epoch}")

# Final test (same style as the first file)
print("\nEvaluating on test set...")
test_recall, test_time, test_edges = test(gnn, test_mp, test_loader, k, device, None, None, batch_size)

test_num_batches   = len(test_loader) if hasattr(test_loader, "__len__") else 0
test_avg_batch_time = test_time / test_num_batches if test_num_batches > 0 else float('inf')
test_edges_per_sec  = test_edges / test_time if test_time > 0 else 0.0
actual_test_bs      = getattr(test_loader, "batch_size", None) or (batch_size * 32)

print(f"Test set recall@{k}: {test_recall:.4f}")
print(f"Test evaluation - Batch size: {actual_test_bs}, Time: {test_time:.2f}s, "
      f"Avg batch time: {test_avg_batch_time:.3f}s, Throughput: {test_edges_per_sec:.0f} edges/sec")

# Log + save metrics like the first file
tracker.log_test_results(test_recall, test_time)
tracker.save()

# Optional: save a final snapshot for reproducibility (parity with feature-aware script)
torch.save({
    'model_state_dict': gnn.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'epoch': epochs,
    'loss': all_train_losses[-1][1] if all_train_losses else None,
    'test_recall': test_recall
}, 'baseline_lightgcn_checkpoint.pt')
print("\nModel saved to baseline_lightgcn_checkpoint.pt")

# Optional: plot train loss (already in your code)