import argparse
import json
import os
import sqlite3
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric import seed_everything


FEATURE_DIM = 7  # Matches training (danceability, energy, speechiness, acousticness, instrumentalness, valence, tempo)


def load_audio_features(db_path: str, song_info: Dict[int, dict], chunk_size: int = 5000) -> Tuple[Dict[int, torch.Tensor], int]:
    """Load and normalize audio features for songs from SQLite database (fast, chunked)."""
    if not db_path:
        return {}, FEATURE_DIM
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found at {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=OFF;")
    cursor.execute("PRAGMA synchronous=OFF;")
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.execute("PRAGMA cache_size=100000;")

    feature_cols = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "valence",
        "tempo",
    ]

    cursor.execute("SELECT MIN(tempo), MAX(tempo) FROM extracted WHERE tempo IS NOT NULL;")
    min_tempo, max_tempo = cursor.fetchone()
    if min_tempo is None or max_tempo is None or max_tempo == min_tempo:
        min_tempo, max_tempo = 0.0, 1.0

    # Map track_id -> node_id
    track_to_node = {info["track_uri"].split(":")[-1]: nid for nid, info in song_info.items()}

    node_to_features: Dict[int, torch.Tensor] = {}
    track_ids = list(track_to_node.keys())
    ph = lambda n: ",".join("?" * n)
    for i in range(0, len(track_ids), chunk_size):
        chunk = track_ids[i : i + chunk_size]
        q = f"SELECT id, {', '.join(feature_cols)} FROM extracted WHERE id IN ({ph(len(chunk))});"
        cursor.execute(q, chunk)
        for row in cursor.fetchall():
            track_id = row[0]
            feats = list(row[1:])
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


def initialize_embeddings_with_features(
    num_nodes: int,
    num_playlists: int,
    emb_dim: int,
    song_features: Dict[int, torch.Tensor],
    feature_dim: int,
) -> nn.Embedding:
    """
    Feature-informed embedding initialization for songs (playlists init randomly),
    matching the training script behavior. The resulting weights will be replaced
    by checkpoint values once loaded, but we must construct the same architecture.
    """
    embeddings = nn.Embedding(num_nodes, emb_dim)
    torch.nn.init.normal_(embeddings.weight[:num_playlists], std=0.1)

    if song_features:
        feature_proj = nn.Linear(feature_dim, emb_dim)
        torch.nn.init.xavier_uniform_(feature_proj.weight)
        with torch.no_grad():
            for node_id, features in song_features.items():
                if node_id < num_nodes:
                    feature_emb = feature_proj(features)
                    random_emb = torch.randn(emb_dim) * 0.1
                    embeddings.weight[node_id] = 0.8 * feature_emb + 0.2 * random_emb
    return embeddings


class FeatureAwareLightGCN(MessagePassing):
    def __init__(self, emb_dim, feature_dim):
        super().__init__(aggr="add")
        # Transform features at each layer
        self.feature_transform = nn.Linear(feature_dim, emb_dim)
        # Learnable gate to balance embeddings and features
        self.gate = nn.Sequential(
            nn.Linear(emb_dim + feature_dim, emb_dim),
            nn.Sigmoid()
        )

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, x, edge_index, edge_weight, node_features):
        # Transform features to embedding space
        feature_emb = self.feature_transform(node_features)

        # Gate mechanism to combine embeddings and features
        combined = torch.cat([x, node_features], dim=1)
        gate_weight = self.gate(combined)

        # Combine original embeddings with transformed features
        x = x + gate_weight * feature_emb

        # Standard LightGCN propagation
        return self.propagate(edge_index, x=x, norm=edge_weight)


class FeatureAwareGNN(torch.nn.Module):
    def __init__(self, emb_dim, num_nodes, num_playlists, layers, song_features, feature_dim):
        super().__init__()
        self.num_playlists = num_playlists
        self.feature_dim = feature_dim

        # Initialize embeddings with features (keep this!)
        self.emb = initialize_embeddings_with_features(
            num_nodes, num_playlists, emb_dim, song_features, feature_dim
        )

        # Feature-aware layers
        self.layers = torch.nn.ModuleList([
            FeatureAwareLightGCN(emb_dim, feature_dim) for _ in range(layers)
        ])

        # Optional: learnable layer importance weights
        self.layer_weights = nn.Parameter(torch.ones(layers + 1) / (layers + 1))

    def gnn_propagation(self, mp: Data, node_features):
        x = self.emb.weight
        outs = [x]

        for layer in self.layers:
            x = layer(x, mp.edge_index, mp.edge_weight, node_features)
            outs.append(x)

        # Weighted combination of layer outputs
        layer_weights = F.softmax(self.layer_weights, dim=0)
        out = torch.stack(outs, dim=0)
        return torch.sum(out * layer_weights.view(-1, 1, 1), dim=0)


def build_mp(data: Data, device: torch.device) -> Data:
    edge_index = data.edge_index.to(device)
    deg = degree(edge_index[1], data.num_nodes, dtype=torch.float32).to(device)
    norm = deg.pow(-0.5)
    weight = norm[edge_index[0]] * norm[edge_index[1]]
    return Data(edge_index=edge_index, edge_weight=weight, num_nodes=data.num_nodes)


def main():
    parser = argparse.ArgumentParser(description="Ready-to-run inference for Feature-Aware Message Passing LightGCN")
    parser.add_argument("--k", type=int, default=300, help="Top-k recommendations (default 300)")
    parser.add_argument("--playlist-id", type=int, default=200, help="Playlist node id to recommend for (0..num_playlists-1). Defaults to 200.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Directory containing data_object.pt, dataset_stats.json, song_info.json, and the checkpoint (defaults to this script's folder).",
    )
    parser.add_argument("--checkpoint", type=str, default="feature_aware_lightgcn_checkpoint.pt", help="Checkpoint filename inside data-dir.")
    parser.add_argument("--audio-db", type=str, default=None, help="Optional path to SQLite audio features DB to use feature-aware message passing at inference.")
    args = parser.parse_args()

    base_dir = args.data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match training seed so the RandomLinkSplit is reproducible
    seed_everything(5)

    # Load graph and stats
    data: Data = torch.load(os.path.join(base_dir, "data_object.pt"), weights_only=False)
    with open(os.path.join(base_dir, "dataset_stats.json")) as f:
        stats = json.load(f)
    with open(os.path.join(base_dir, "song_info.json")) as f:
        song_info = json.load(f)
        song_info = {int(k): v for k, v in song_info.items()}

    num_playlists, num_nodes = stats["num_playlists"], stats["num_nodes"]

    # Optional: load audio features from SQLite DB; otherwise use zeros
    song_features = {}
    feature_dim = FEATURE_DIM
    if args.audio_db:
        song_features, feature_dim = load_audio_features(args.audio_db, song_info)
    feature_matrix = torch.zeros(num_nodes, feature_dim, dtype=torch.float32)
    for node_id, feats in song_features.items():
        if node_id < num_nodes:
            feature_matrix[node_id] = feats
    feature_matrix = feature_matrix.to(device)

    # Build train/val/test split to (a) avoid leakage in propagation, (b) recover the hidden edges for evaluation
    split = RandomLinkSplit(
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0,
        num_val=0.15,
        num_test=0.15,
    )
    train_split, _val_split, test_split = split(data)
    mp = build_mp(train_split, device)  # normalized graph from train edges only

    # Recreate model architecture (match training hyperparams) and load checkpoint
    emb_dim, num_layers = 64, 3
    model = FeatureAwareGNN(
        emb_dim=emb_dim,
        num_nodes=num_nodes,
        num_playlists=num_playlists,
        layers=num_layers,
        song_features={},  # weights overwritten by checkpoint (keeps shapes consistent)
        feature_dim=feature_dim,
    ).to(device)
    ckpt_path = os.path.join(base_dir, args.checkpoint)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()

    # Propagate to get final embeddings
    with torch.no_grad():
        embs = model.gnn_propagation(mp, feature_matrix)

    # Choose a playlist to recommend for (default 200)
    pl = max(0, min(int(args.playlist_id), num_playlists - 1))

    # Score songs for the selected playlist and exclude already-known items
    pl_emb, song_emb = embs[:num_playlists], embs[num_playlists:]
    scores = (pl_emb[pl : pl + 1] @ song_emb.T).squeeze(0)
    # Mask only training-known edges (to mirror evaluation protocol)
    train_edge_index = train_split.edge_index.to(device)
    mask_known = train_edge_index[:, train_edge_index[0] == pl]
    known_song_ids = mask_known[1] - num_playlists
    scores[known_song_ids] = -1e9
    top_vals, top_idx = torch.topk(scores, k=args.k)

    # Pretty-print recommendations
    song_node_ids = (top_idx + num_playlists).tolist()
    print("\n=== Inference (Feature-Aware Message Passing) ===")
    print(f"Device: {device}")
    print(f"Top-{args.k} recommendations for playlist {pl}:")
    for rank, nid in enumerate(song_node_ids, start=1):
        meta = song_info.get(nid, {})
        t = meta.get("track_name", f"<song {nid}>")
        a = meta.get("artist_name", "<artist>")
        print(f"  {rank:2d}. {t} — {a} (node {nid})")

    # Compute recall@k for this specific playlist using the hidden test edges
    gt = test_split.edge_label_index.to(device)
    mask_pl = gt[0] == pl
    pos_songs = gt[1, mask_pl]
    total_hidden = int(pos_songs.numel())
    if total_hidden > 0:
        rec_song_nodes = torch.tensor(song_node_ids, device=device)
        hits = int((pos_songs[:, None] == rec_song_nodes).any(1).sum().item())
        recall_k = hits / total_hidden
        print(f"\nHidden edges for playlist {pl} in test split: {total_hidden}")
        print(f"Hits in top-{args.k}: {hits}")
        print(f"Recall@{args.k} for playlist {pl}: {recall_k:.4f}")
    else:
        print(f"\nNo hidden test edges for playlist {pl}; cannot compute recall.")


if __name__ == "__main__":
    # Allow running via `python FeatureAwareMessagePassing/inference.py`
    main()
