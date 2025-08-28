import json, os, time, numpy as np, torch, matplotlib.pyplot as plt, sqlite3
from torch_geometric import seed_everything
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import degree

# ----------------------------------------------------------------------
# AMP helpers (support new & old API names)
# ----------------------------------------------------------------------
try:                                 # PyTorch â‰¥2.3
    from torch.amp import autocast, GradScaler
    AMP_KW = dict(device_type="cuda", dtype=torch.float16)
except ImportError:                  # PyTorch â‰¤2.2
    from torch.cuda.amp import autocast, GradScaler
    AMP_KW = {}

# ----------------------------------------------------------------------
# Load audio features from SQLite  (FIXED: bulk, chunked fetching)
# ----------------------------------------------------------------------
def load_audio_features(db_path, song_info, chunk_size=5000):
    """Load and normalize audio features for songs from SQLite database quickly."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found at {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Ensure fast lookups
    cursor.execute("PRAGMA journal_mode=OFF;")
    cursor.execute("PRAGMA synchronous=OFF;")
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.execute("PRAGMA cache_size=100000;")  # ~100MB

    feature_cols = ['danceability', 'energy', 'speechiness', 'acousticness',
                    'instrumentalness', 'valence', 'tempo']

    # Tempo stats (could use global stats; it's fine)
    cursor.execute("SELECT MIN(tempo), MAX(tempo) FROM extracted WHERE tempo IS NOT NULL;")
    min_tempo, max_tempo = cursor.fetchone()
    if min_tempo is None or max_tempo is None or max_tempo == min_tempo:
        min_tempo, max_tempo = 0.0, 1.0  # avoid div-by-zero

    # Map track_id -> node_id
    track_to_node = {info['track_uri'].split(':')[-1]: nid for nid, info in song_info.items()}

    # Pull in chunks
    track_ids = list(track_to_node.keys())
    node_to_features = {}

    placeholders = lambda n: ",".join("?" * n)
    for i in range(0, len(track_ids), chunk_size):
        chunk = track_ids[i:i + chunk_size]
        q = f"SELECT id, {', '.join(feature_cols)} FROM extracted WHERE id IN ({placeholders(len(chunk))});"
        cursor.execute(q, chunk)
        for row in cursor.fetchall():
            track_id = row[0]
            feats = list(row[1:])
            # Normalize tempo (index 6)
            if feats[6] is not None:
                feats[6] = (feats[6] - min_tempo) / (max_tempo - min_tempo + 1e-8)
            else:
                feats[6] = 0.5
            feats = [f if f is not None else 0.5 for f in feats]
            node_id = track_to_node.get(track_id)
            if node_id is not None:
                node_to_features[node_id] = torch.tensor(feats, dtype=torch.float32)

    conn.close()
    return node_to_features, len(feature_cols)

# ----------------------------------------------------------------------
# Feature-informed embedding initialization
# ----------------------------------------------------------------------
def initialize_embeddings_with_features(num_nodes, num_playlists, emb_dim, song_features, feature_dim):
    """Initialize embeddings using audio features for songs."""
    embeddings = torch.nn.Embedding(num_nodes, emb_dim)

    # Initialize playlist embeddings normally
    torch.nn.init.normal_(embeddings.weight[:num_playlists], std=0.1)

    # For songs with features, use feature-informed initialization
    if song_features:
        feature_proj = torch.nn.Linear(feature_dim, emb_dim)
        torch.nn.init.xavier_uniform_(feature_proj.weight)

        with torch.no_grad():
            for node_id, features in song_features.items():
                if node_id < num_nodes:
                    feature_emb = feature_proj(features)
                    random_emb = torch.randn(emb_dim) * 0.1
                    embeddings.weight[node_id] = 0.8 * feature_emb + 0.2 * random_emb

    return embeddings

# ----------------------------------------------------------------------
# Optional: popularity-based song distribution for GPU negative sampling
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Setup & data
# ----------------------------------------------------------------------
seed_everything(5)
base_dir = "."
data = torch.load(os.path.join(base_dir, "data_object.pt"), weights_only=False)
with open(os.path.join(base_dir, "dataset_stats.json")) as f:
    stats = json.load(f)
with open(os.path.join(base_dir, "song_info.json")) as f:
    song_info = json.load(f)
    song_info = {int(k): v for k, v in song_info.items()}

num_playlists, num_nodes = stats["num_playlists"], stats["num_nodes"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable cudnn benchmark if available
if hasattr(torch.backends, "cudnn"):
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# Load audio features
print("Loading audio features from SQLite...")
song_features, feature_dim = load_audio_features("extracted.db", song_info)
print(f"Loaded features for {len(song_features)} songs out of {len(song_info)} total songs")

transform = RandomLinkSplit(
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=0,
    num_val=0.15,
    num_test=0.15,
)
train_split, val_split, test_split = transform(data)

# ----------------------------------------------------------------------
# Build GPU-resident graphs
# ----------------------------------------------------------------------
def build_mp(split: Data) -> Data:
    edge_index = split.edge_index.to(device)
    deg = degree(edge_index[1], data.num_nodes, dtype=torch.float32).to(device)
    norm = deg.pow(-0.5)
    weight = norm[edge_index[0]] * norm[edge_index[1]]
    return Data(edge_index=edge_index, edge_weight=weight, num_nodes=data.num_nodes)

train_mp, val_mp, test_mp = map(build_mp, (train_split, val_split, test_split))

# Precompute song popularity probs for optional popularity-based GPU negatives
song_pop_probs = build_song_popularity_probs(train_split.edge_index, num_playlists, num_nodes, device)

# ----------------------------------------------------------------------
# Dataset & DataLoader
# ----------------------------------------------------------------------
class PlainData(Data):
    def __inc__(self, *_): return 0

class SpotifyDataset(Dataset):
    def __init__(self, root: str, edges: torch.Tensor):
        self.edge_index = edges
        self.unique = torch.unique(edges[0]).tolist()
        super().__init__(root)
    def len(self): return len(self.unique)
    def get(self, idx):
        pl = self.unique[idx]
        ei = self.edge_index[:, self.edge_index[0] == pl]
        d = PlainData(edge_index=ei)
        d.num_nodes = int(ei.max()) + 1
        return d

# Training hyperparameters
epochs = 45
k = 300
batch_size = 512

# Negative sampling mode
NEG_SAMPLING_MODE = "uniform"  # change to "popularity" to enable popularity-weighted GPU sampling

# AMP settings with bfloat16 support check
use_amp = torch.cuda.is_available()
amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

# Update AMP_KW based on dtype selection
if use_amp:
    AMP_KW = dict(device_type="cuda", dtype=amp_dtype)

scaler = GradScaler(enabled=use_amp)

# Print all hyperparameters
print("\n" + "="*60)
print("HYPERPARAMETERS")
print("="*60)
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: 1e-3")
print(f"Optimizer: Adam")
print(f"Embedding dimension: 64")
print(f"Number of LightGCN layers: 3")
print(f"k for recall@k: {k}")
print(f"Device: {device}")
print(f"Loss function: BPR (Bayesian Personalized Ranking)")
print(f"Negative sampling ratio: 1:1")
print(f"Negative sampling mode: {NEG_SAMPLING_MODE}")
print(f"Feature-informed initialization: Enabled")
print(f"Number of features: {feature_dim}")
print(f"AMP enabled: {use_amp}")
print(f"AMP dtype: {amp_dtype if use_amp else 'N/A'}")
print("="*60)

# Optimized DataLoader settings
loader_kw = dict(pin_memory=True, num_workers=14, prefetch_factor=16, persistent_workers=True)
train_loader = DataLoader(SpotifyDataset("tmp", train_split.edge_label_index),
                          batch_size=batch_size, shuffle=True, **loader_kw)
val_loader   = DataLoader(SpotifyDataset("tmp", val_split.edge_label_index),
                          batch_size=batch_size*4, shuffle=False, **loader_kw)
def test_loader_fn():
    return DataLoader(SpotifyDataset("tmp", test_split.edge_label_index),
                      batch_size=batch_size*4, shuffle=False, **loader_kw)

# ----------------------------------------------------------------------
# Model with feature-informed initialization
# ----------------------------------------------------------------------
class LightGCN(MessagePassing):
    def __init__(self): super().__init__(aggr="add")
    def message(self, x_j, norm): return norm.view(-1,1)*x_j
    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, norm=edge_weight)

class GNN(torch.nn.Module):
    def __init__(self, emb_dim, num_nodes, num_playlists, layers, song_features=None, feature_dim=None):
        super().__init__()
        self.num_playlists = num_playlists
        self.emb = initialize_embeddings_with_features(
            num_nodes, num_playlists, emb_dim, song_features, feature_dim
        )
        self.layers = torch.nn.ModuleList([LightGCN() for _ in range(layers)])

    def gnn_propagation(self, mp: Data):
        x = self.emb.weight.to(device)
        outs = [x]
        for layer in self.layers:
            x = layer(x, mp.edge_index, mp.edge_weight)
            outs.append(x)
        return torch.stack(outs).mean(0)

    @staticmethod
    def predict(edge_idx, embs):
        h_i, h_j = embs[edge_idx[0]], embs[edge_idx[1]]
        return torch.sigmoid((h_i*h_j).sum(1))

    def calc_loss(self, mp, pos, neg):
        embs = self.gnn_propagation(mp)
        ps  = self.predict(pos.edge_index.to(device), embs)
        ns  = self.predict(neg.edge_index.to(device), embs)
        return -torch.log(torch.sigmoid(ps-ns)).mean()

    def evaluation(self, mp, pos, k):
        embs = self.gnn_propagation(mp)
        pl_emb, song_emb = embs[:self.num_playlists], embs[self.num_playlists:]
        ratings = torch.sigmoid(pl_emb @ song_emb.T)
        return gpu_recall(mp.edge_index,
                          ratings,
                          pos.edge_index.to(device),
                          self.num_playlists, k)

# ----------------------------------------------------------------------
# GPU recall@k
# ----------------------------------------------------------------------
def gpu_recall(edge_index, ratings, ground_truth, num_playlists, k):
    """
    All tensors must already reside on CUDA.  Returns a Python float.
    """
    row, col = edge_index
    pl_mask = row < num_playlists
    mask = torch.zeros_like(ratings, dtype=torch.bool)
    mask[row[pl_mask], col[pl_mask] - num_playlists] = True
    scores = ratings.masked_fill(mask, -1e4)
    _, topk = torch.topk(scores, k, 1)
    topk += num_playlists

    unique_pl = torch.unique(ground_truth[0]).cpu().tolist()
    recalls = []
    for pl in unique_pl:
        pos = ground_truth[1][ground_truth[0] == pl]
        if pos.numel() == 0:
            recalls.append(torch.tensor(0., device=device))
            continue
        rec_vec = topk[pl]
        hits = (pos[:, None] == rec_vec).any(1).float().sum()
        recalls.append(hits / pos.size(0))
    return float(torch.stack(recalls).mean().item())

# ----------------------------------------------------------------------
# GPU-based negative edge sampling (vectorized)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------
def train(model, mp, loader, opt):
    model.train()
    tot_loss = 0
    tot_edges = 0
    epoch_start_time = time.time()
    
    for batch in loader:
        opt.zero_grad(set_to_none=True)
        batch = batch.to(device)
        neg = sample_negative_edges_gpu(batch, num_playlists, num_nodes, device,
                                       negative_mode=NEG_SAMPLING_MODE, song_pop_probs=song_pop_probs)
        
        if use_amp and scaler is not None:
            with autocast(**AMP_KW):
                loss = model.calc_loss(mp, batch, neg)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss = model.calc_loss(mp, batch, neg)
            loss.backward()
            opt.step()
        
        n = batch.edge_index.size(1)
        tot_loss += loss.item()*n
        tot_edges += n
    
    epoch_time = time.time() - epoch_start_time
    return tot_loss/tot_edges, tot_edges, epoch_time

@torch.no_grad()
def evaluate(model, mp, loader, k):
    model.eval()
    recs = []
    eval_start_time = time.time()
    total_edges = 0
    
    with autocast(**AMP_KW) if use_amp else torch.cuda.amp.autocast(enabled=False):
        for batch in loader:
            batch = batch.to(device)
            recs.append(model.evaluation(mp, batch, k))
            total_edges += batch.edge_index.size(1)
    
    eval_time = time.time() - eval_start_time
    return float(np.mean(recs)), eval_time, total_edges

# ----------------------------------------------------------------------
# Main loop with feature-informed model
# ----------------------------------------------------------------------
model = GNN(emb_dim=64, num_nodes=data.num_nodes,
            num_playlists=num_playlists, layers=3,
            song_features=song_features, feature_dim=feature_dim).to(device)

# Torch compile with better mode
try:
    model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    print("\ntorch.compile enabled with mode='reduce-overhead'")
except Exception as e:
    print(f"\ntorch.compile not available or failed ({e}); continuing without compile.")

# Count and print model parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal trainable parameters: {total_params:,}")
print("="*60 + "\n")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
history = []

print("Starting training with feature-informed initialization...")
for epoch in range(epochs):
    tr_loss, train_edges, train_time = train(model, train_mp, train_loader, opt)
    history.append((epoch, tr_loss))
    
    # Calculate training metrics
    num_batches = len(train_loader)
    avg_batch_time = train_time / num_batches if num_batches > 0 else float('inf')
    edges_per_sec = train_edges / train_time if train_time > 0 else 0.0

    if epoch >= 40:
        val_rec, val_time, val_edges = evaluate(model, val_mp, val_loader, k)
        
        # Calculate validation metrics
        val_num_batches = len(val_loader)
        val_avg_batch_time = val_time / val_num_batches if val_num_batches > 0 else float('inf')
        val_edges_per_sec = val_edges / val_time if val_time > 0 else 0.0
        
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f}, val_recall={val_rec:.4f}")
        print(f"  Training   - Batch size: {batch_size}, Time: {train_time:.2f}s, Avg batch time: {avg_batch_time:.3f}s, Throughput: {edges_per_sec:.0f} edges/sec")
        print(f"  Validation - Batch size: {batch_size*4}, Time: {val_time:.2f}s, Avg batch time: {val_avg_batch_time:.3f}s, Throughput: {val_edges_per_sec:.0f} edges/sec")
    else:
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f}")
        print(f"  Training   - Batch size: {batch_size}, Time: {train_time:.2f}s, Avg batch time: {avg_batch_time:.3f}s, Throughput: {edges_per_sec:.0f} edges/sec")

# ----------------------------------------------------------------------
# Final test
# ----------------------------------------------------------------------
print("\nEvaluating on test set...")
test_loader = test_loader_fn()
test_rec, test_time, test_edges = evaluate(model, test_mp, test_loader, k)

# Calculate test metrics
test_num_batches = len(test_loader)
test_avg_batch_time = test_time / test_num_batches if test_num_batches > 0 else float('inf')
test_edges_per_sec = test_edges / test_time if test_time > 0 else 0.0

print(f"Test set recall@{k}: {test_rec:.4f}")
print(f"Test evaluation - Batch size: {batch_size*4}, Time: {test_time:.2f}s, Avg batch time: {test_avg_batch_time:.3f}s, Throughput: {test_edges_per_sec:.0f} edges/sec")

# ----------------------------------------------------------------------
# Plot loss
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(*zip(*history))
plt.xlabel("Epoch")
plt.ylabel("Train loss")
plt.title("Loss over time (Feature-Informed LightGCN)")
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# Save the model
# ----------------------------------------------------------------------
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'epoch': epochs,
    'loss': history[-1][1] if history else None,
    'test_recall': test_rec
}, 'lightgcn_with_features_checkpoint.pt')
print("\nModel saved to lightgcn_with_features_checkpoint.pt")