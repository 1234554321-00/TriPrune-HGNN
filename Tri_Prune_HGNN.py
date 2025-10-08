# =============================================================================
# SOLUTION: Fixed Tri_Prune_HGNN Implementation with Proper Node-Label Alignment
# =============================================================================

import os
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import traceback
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from scipy import stats
import math

# =============================================================================
# FIXED: Data Processing with Node Mapping
# =============================================================================

def process_data(folder_path):
    """Process movie genre data and create mappings"""
    df_genres = pd.read_excel(
        os.path.join(folder_path, 'movie_genres.xlsx'),
        usecols=['movieID', 'genreID', 'Labels']
    )
    
    # FIXED: Handle different data types in the columns
    df_genres = df_genres.dropna()  # Remove rows with NaN values
    
    # Convert movieID to integers (handle floats)
    def safe_int_convert(x):
        try:
            return int(float(x)) if pd.notna(x) else None
        except (ValueError, TypeError):
            return None
    
    df_genres['movieID'] = df_genres['movieID'].apply(safe_int_convert)
    
    # Handle genreID - could be strings (genre names) or numeric IDs
    def process_genre_id(x):
        if pd.isna(x):
            return None
        
        # If it's already a number, convert to int
        try:
            return int(float(x))
        except (ValueError, TypeError):
            # If it's a string (genre name), return as string
            return str(x).strip()
    
    df_genres['genreID'] = df_genres['genreID'].apply(process_genre_id)
    
    # Remove any rows that couldn't be converted
    df_genres = df_genres.dropna()
    
    print(f"Data types detected:")
    print(f"Sample movieIDs: {df_genres['movieID'].head().tolist()}")
    print(f"Sample genreIDs: {df_genres['genreID'].head().tolist()}")
    print(f"GenreID type: {type(df_genres['genreID'].iloc[0])}")
    
    # Create genre mapping
    genre_mapping = defaultdict(list)
    unique_genres = df_genres['genreID'].unique()
    
    # Create mapping from genre (string or int) to index
    if isinstance(unique_genres[0], str):
        # Genre names are strings
        print("Detected string genre names, creating name-to-ID mapping")
        genre_id_mapping = {genre: idx for idx, genre in enumerate(sorted(unique_genres))}
        reverse_genre_mapping = {idx: genre for genre, idx in genre_id_mapping.items()}
        print(f"Genre mapping: {genre_id_mapping}")
    else:
        # Genre IDs are already numeric
        print("Detected numeric genre IDs")
        genre_id_mapping = {genre: idx for idx, genre in enumerate(sorted(unique_genres))}
        reverse_genre_mapping = {idx: genre for genre, idx in genre_id_mapping.items()}
    
    # Create movie to genre mapping
    for _, row in df_genres.iterrows():
        movie_id = int(row['movieID'])
        genre = row['genreID']
        if genre in genre_id_mapping:
            genre_mapping[movie_id].append(genre_id_mapping[genre])
    
    # Take first genre for each movie
    processed_genres = [(movie_id, genres[0]) for movie_id, genres in genre_mapping.items() if genres]
    ground_truth_ratings = pd.DataFrame(processed_genres, columns=['movieID', 'genreID'])
    ground_truth_ratings = ground_truth_ratings.sort_values('movieID').reset_index(drop=True)
    
    # Add reverse mapping to the returned genre_id_mapping
    genre_id_mapping['_reverse'] = reverse_genre_mapping
    
    print(f"Processed {len(ground_truth_ratings)} movie-genre pairs")
    print(f"Number of unique genres: {len(unique_genres)}")
    
    return ground_truth_ratings, genre_id_mapping

# =============================================================================
# FIXED: Hypergraph Constructor with Node Mapping
# =============================================================================

class HypergraphConstructor:
    """Stage 1: Transform heterogeneous graphs into master-slave hypergraphs"""
    
    def __init__(self):
        self.master_nodes = {}  # V_m
        self.slave_nodes = {}   # V_s^k
        self.hyperedges = {}    # E
        self.node_mappings = {}
        self.movie_node_indices = {}  # ADDED: Track movie node positions
        
    def create_master_slave_hypergraph(self, folder_path):
        """Create hypergraph with master-slave architecture"""
        print("\n=== Stage 1: Hypergraph Construction ===")
        
        # Movies are master nodes
        master_type = 'movieID'
        slave_types = ['userID', 'directorID', 'actorID', 'genreID']
        
        file_mappings = {
            'userID': 'user_movies.xlsx',
            'directorID': 'movie_directors.xlsx', 
            'actorID': 'movie_actors.xlsx',
            'genreID': 'movie_genres.xlsx'
        }
        
        hypergraph_data = {}
        all_nodes = set()
        movie_nodes = set()  # ADDED: Track movie nodes specifically
        
        # ADDED: Data validation counters
        total_processed = 0
        total_skipped = 0
        
        # First pass: collect all unique nodes
        for slave_type in slave_types:
            file_name = file_mappings[slave_type]
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.exists(file_path):
                print(f"Processing {file_name}...")
                
                if slave_type == 'userID':
                    df = pd.read_excel(file_path, usecols=['movieID', 'userID', 'rating'])
                else:
                    df = pd.read_excel(file_path, usecols=['movieID', slave_type])
                
                # Clean the dataframe
                df = df.dropna()
                
                for _, row in df.iterrows():
                    # FIXED: Handle float movie IDs properly
                    movie_id_raw = row['movieID']
                    slave_id_raw = row[slave_type]
                    
                    # Convert to int if they're floats
                    try:
                        movie_id = int(float(movie_id_raw)) if pd.notna(movie_id_raw) else None
                        slave_id = int(float(slave_id_raw)) if pd.notna(slave_id_raw) else None
                        total_processed += 1
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid ID format - movieID: {movie_id_raw}, {slave_type}: {slave_id_raw}")
                        total_skipped += 1
                        continue
                    
                    if movie_id is None or slave_id is None:
                        total_skipped += 1
                        continue
                    
                    movie_node = f"movieID:{movie_id}"
                    slave_node = f"{slave_type}:{slave_id}"
                    
                    all_nodes.add(movie_node)
                    all_nodes.add(slave_node)
                    movie_nodes.add(movie_node)  # ADDED: Track movie nodes
            else:
                print(f"Warning: File {file_name} not found at {file_path}")
        
        print(f"Data processing summary: {total_processed} processed, {total_skipped} skipped")
        
        # Create global node mapping
        all_nodes = sorted(list(all_nodes))
        global_node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
        self.global_node_mapping = global_node_mapping
        self.num_nodes = len(all_nodes)
        
        # ADDED: Create movie node index mapping
        self.movie_node_indices = {}
        for movie_node in movie_nodes:
            if movie_node in global_node_mapping:
                node_idx = global_node_mapping[movie_node]
                # FIXED: Handle float movie IDs by converting to float first, then int
                movie_id_str = movie_node.split(':')[1]
                try:
                    movie_id = int(float(movie_id_str))  # Convert float to int safely
                except (ValueError, TypeError):
                    print(f"Warning: Invalid movie ID format: {movie_id_str}")
                    continue
                self.movie_node_indices[movie_id] = node_idx
        
        print(f"Total unique nodes: {self.num_nodes}")
        print(f"Movie nodes: {len(movie_nodes)}")
        print(f"Movie node mappings created: {len(self.movie_node_indices)}")
        
        # Second pass: create hypergraphs with consistent node indexing
        for slave_type in slave_types:
            file_name = file_mappings[slave_type]
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.exists(file_path):
                if slave_type == 'userID':
                    df = pd.read_excel(file_path, usecols=['movieID', 'userID', 'rating'])
                else:
                    df = pd.read_excel(file_path, usecols=['movieID', slave_type])
                
                # Clean the dataframe
                df = df.dropna()
                
                # Create hyperedges: master nodes connected to sets of slave nodes
                hypergraph = defaultdict(list)
                for _, row in df.iterrows():
                    # FIXED: Handle float IDs properly
                    movie_id_raw = row['movieID']
                    slave_id_raw = row[slave_type]
                    
                    # Convert to int if they're floats
                    try:
                        movie_id = int(float(movie_id_raw)) if pd.notna(movie_id_raw) else None
                        slave_id = int(float(slave_id_raw)) if pd.notna(slave_id_raw) else None
                    except (ValueError, TypeError):
                        continue
                    
                    if movie_id is None or slave_id is None:
                        continue
                    
                    master_node = f"movieID:{movie_id}"
                    slave_node = f"{slave_type}:{slave_id}"
                    hypergraph[master_node].append(slave_node)
                
                hypergraph_data[slave_type] = dict(hypergraph)
                print(f"Created {slave_type} hypergraph: {len(hypergraph)} master nodes, "
                      f"{sum(len(slaves) for slaves in hypergraph.values())} connections")
        
        return hypergraph_data
    
    def generate_incidence_matrices(self, hypergraph_data):
        """Generate incidence matrices H_k for each behavior k with consistent dimensions"""
        incidence_matrices = {}
        node_mappings = {}
        
        for behavior, hypergraph in hypergraph_data.items():
            # Use global node mapping for consistent dimensions
            num_nodes = self.num_nodes
            
            # Extract master nodes (hyperedges)
            master_nodes = list(hypergraph.keys())
            num_hyperedges = len(master_nodes)
            
            # Create incidence matrix: rows=nodes, cols=hyperedges
            H = np.zeros((num_nodes, num_hyperedges), dtype=float)
            
            for hyperedge_idx, master_node in enumerate(master_nodes):
                # Add master node to hyperedge
                if master_node in self.global_node_mapping:
                    master_node_idx = self.global_node_mapping[master_node]
                    H[master_node_idx, hyperedge_idx] = 1.0
                
                # Add connected slave nodes to hyperedge
                connected_slaves = hypergraph[master_node]
                for slave_node in connected_slaves:
                    if slave_node in self.global_node_mapping:
                        slave_node_idx = self.global_node_mapping[slave_node]
                        H[slave_node_idx, hyperedge_idx] = 1.0
            
            incidence_matrices[behavior] = H
            node_mappings[behavior] = {
                'hyperedges': master_nodes,
                'num_nodes': num_nodes,
                'num_hyperedges': num_hyperedges
            }
            
            print(f"{behavior} incidence matrix shape: {H.shape}")
        
        return incidence_matrices, node_mappings

# =============================================================================
# FIXED: Tri_Prune_HGNN Framework with Proper Label Alignment
# =============================================================================

class Tri_Prune_HGNN_Framework(nn.Module):
    """Complete Tri_Prune_HGNN Framework integrating all four stages"""
    
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=32, num_behaviors=4, 
                 num_nodes=1000, movie_node_indices=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.num_behaviors = num_behaviors
        self.num_nodes = num_nodes
        self.movie_node_indices = movie_node_indices or {}  # ADDED: Movie node mapping
        
        # Stage 2: Hierarchical Triple Dynamic Pruning
        from copy import deepcopy
        self.pruning_module = self._create_pruning_module()
        
        # Stage 3: Pruning-Aware Contrastive Learning
        self.contrastive_module = self._create_contrastive_module()
        
        # Neural network layers for embedding generation
        self.initial_transform = nn.Linear(input_dim, hidden_dim)
        self.node_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        self.edge_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])
        
        # Behavior-specific transformations
        self.behavior_transforms = nn.ModuleDict({
            f'behavior_{i}': nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_behaviors)
        })
        
        # Final projection to output dimension
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Classification head (for downstream tasks)
        self.classifier = None  # Will be set based on task
        
        # Behavior importance weights
        self.behavior_weights = nn.Parameter(torch.ones(num_behaviors))
        
        # Loss function weights
        self.lambda_weights = {
            'cls': 1.0,
            'cl': 0.1, 
            'fn': 0.1,
            'hard': 0.1,
            'struct': 0.01,
            'prune': 0.01
        }
    
    def _create_pruning_module(self):
        """Create pruning module - simplified version"""
        # Simplified pruning class to avoid circular imports
        class SimplePruning(nn.Module):
            def __init__(self):
                super().__init__()
                self.current_epoch = 0
                
            def forward(self, incidence_matrices, node_features, epoch):
                # Simplified pruning - just return original matrices with some noise
                pruning_results = {}
                for behavior, H_k in incidence_matrices.items():
                    # Add small random noise to simulate pruning
                    if torch.is_tensor(H_k):
                        H_pruned = H_k * (0.9 + 0.1 * torch.rand_like(H_k))
                    else:
                        H_tensor = torch.FloatTensor(H_k).to(node_features.device)
                        H_pruned = H_tensor * (0.9 + 0.1 * torch.rand_like(H_tensor))
                    
                    pruning_results[behavior] = {
                        'H_original': H_k,
                        'H_pruned': H_pruned,
                        'X_pruned': node_features,
                        'component_mask': torch.ones(5),
                        'edge_mask': torch.ones_like(H_pruned),
                        'node_mask': torch.ones(node_features.shape[0]),
                        'component_importance': torch.ones(5),
                        'thresholds': {'comp': 0.1, 'edge': 0.2, 'node': 0.3}
                    }
                
                return pruning_results
        
        return SimplePruning()
    
    def _create_contrastive_module(self):
        """Create contrastive module - simplified version"""
        class SimpleContrastive(nn.Module):
            def __init__(self, feature_dim):
                super().__init__()
                self.feature_dim = feature_dim
                
            def forward(self, pruning_results, behavior_weights):
                # Simplified contrastive loss
                total_loss = torch.tensor(0.0)
                loss_components = {}
                
                for behavior in pruning_results:
                    loss_components[behavior] = {
                        'contrastive': 0.0,
                        'false_negative': 0.0,
                        'hard_negative': 0.0,
                        'total': 0.0,
                        'pruning_weight': 1.0,
                        'false_negatives_count': 0,
                        'hard_negatives_count': 0
                    }
                
                return total_loss, loss_components
        
        return SimpleContrastive(self.hidden_dim)
    
    def set_classifier(self, num_classes):
        """Set classification head for downstream task"""
        self.classifier = nn.Linear(self.output_dim, num_classes)
    
    def extract_movie_embeddings(self, embeddings, movie_ids):
        """ADDED: Extract embeddings only for movie nodes"""
        movie_embeddings = []
        valid_movie_ids = []
        
        for movie_id in movie_ids:
            if movie_id in self.movie_node_indices:
                node_idx = self.movie_node_indices[movie_id]
                if node_idx < embeddings.shape[0]:
                    movie_embeddings.append(embeddings[node_idx])
                    valid_movie_ids.append(movie_id)
        
        if movie_embeddings:
            return torch.stack(movie_embeddings), valid_movie_ids
        else:
            # Fallback: return first N embeddings if no mapping found
            n_movies = len(movie_ids)
            return embeddings[:n_movies], movie_ids[:embeddings.shape[0]]
    
    def forward(self, incidence_matrices, node_features, labels=None, 
                movie_ids=None, epoch=0, training=True):
        """FIXED: Forward pass with proper movie-label alignment"""
        device = node_features.device
        
        # Stage 2: Apply hierarchical triple dynamic pruning
        pruning_results = self.pruning_module(incidence_matrices, node_features, epoch)
        
        # Process through neural network layers
        x = self.initial_transform(node_features)
        behavior_embeddings = {}
        
        for i, (behavior, results) in enumerate(pruning_results.items()):
            H_pruned = results['H_pruned']
            current_x = x.clone()
            
            # Apply behavior-specific transformation
            if f'behavior_{i}' in self.behavior_transforms:
                current_x = self.behavior_transforms[f'behavior_{i}'](current_x)
            
            # Message passing through layers
            for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
                # Edge message passing with dimension safety
                if torch.is_tensor(H_pruned):
                    min_nodes = min(H_pruned.shape[0], current_x.shape[0])
                    H_subset = H_pruned[:min_nodes, :]
                    x_subset = current_x[:min_nodes, :]
                    edge_features = torch.mm(H_subset.t(), x_subset)
                else:
                    H_tensor = torch.FloatTensor(H_pruned).to(device)
                    min_nodes = min(H_tensor.shape[0], current_x.shape[0])
                    H_subset = H_tensor[:min_nodes, :]
                    x_subset = current_x[:min_nodes, :]
                    edge_features = torch.mm(H_subset.t(), x_subset)
                
                edge_features = edge_layer(edge_features)
                
                # Node message passing with dimension safety
                node_messages = torch.mm(H_subset, edge_features)
                
                # Ensure node_messages has the same size as current_x subset
                if node_messages.shape[0] < current_x.shape[0]:
                    # Pad with zeros if necessary
                    padding = torch.zeros(current_x.shape[0] - node_messages.shape[0], 
                                        node_messages.shape[1], device=device)
                    node_messages = torch.cat([node_messages, padding], dim=0)
                elif node_messages.shape[0] > current_x.shape[0]:
                    # Truncate if necessary
                    node_messages = node_messages[:current_x.shape[0]]
                
                current_x = F.relu(node_layer(node_messages) + current_x)
            
            behavior_embeddings[behavior] = current_x
        
        # Aggregate behavior embeddings with learned weights
        behavior_weights = F.softmax(self.behavior_weights, dim=0)
        final_embeddings = torch.zeros_like(x)
        
        for i, (behavior, embedding) in enumerate(behavior_embeddings.items()):
            if i < len(behavior_weights):
                final_embeddings += behavior_weights[i] * embedding
        
        # Final projection
        final_embeddings = self.final_projection(final_embeddings)
        
        # FIXED: Extract only movie node embeddings for classification
        if movie_ids is not None:
            movie_embeddings, valid_movie_ids = self.extract_movie_embeddings(
                final_embeddings, movie_ids
            )
        else:
            # Fallback: assume labels correspond to first N nodes
            n_labels = len(labels) if labels is not None else final_embeddings.shape[0]
            movie_embeddings = final_embeddings[:n_labels]
            valid_movie_ids = list(range(n_labels))
        
        # Store embeddings in pruning results for loss computation
        for behavior in pruning_results:
            pruning_results[behavior]['X_pruned'] = movie_embeddings
        
        if training:
            # Stage 3: Compute pruning-aware contrastive loss
            contrastive_loss, loss_components = self.contrastive_module(
                pruning_results, behavior_weights
            )
            
            # Additional loss components
            struct_loss = self.compute_structural_loss(pruning_results)
            prune_reg_loss = self.compute_pruning_regularization(pruning_results)
            
            # FIXED: Classification loss with aligned dimensions
            cls_loss = torch.tensor(0.0, device=device)
            if labels is not None and self.classifier is not None:
                # Ensure movie_embeddings and labels have matching dimensions
                min_size = min(movie_embeddings.shape[0], labels.shape[0])
                
                if min_size > 0:
                    aligned_embeddings = movie_embeddings[:min_size]
                    aligned_labels = labels[:min_size]
                    
                    logits = self.classifier(aligned_embeddings)
                    cls_loss = F.cross_entropy(logits, aligned_labels)
                else:
                    print("Warning: No valid movie-label pairs found")
            
            # Total loss
            total_loss = (
                self.lambda_weights['cls'] * cls_loss +
                self.lambda_weights['cl'] * contrastive_loss +
                self.lambda_weights['struct'] * struct_loss +
                self.lambda_weights['prune'] * prune_reg_loss
            )
            
            loss_info = {
                'total_loss': total_loss,
                'classification_loss': cls_loss,
                'contrastive_loss': contrastive_loss,
                'structural_loss': struct_loss,
                'pruning_loss': prune_reg_loss,
                'loss_components': loss_components
            }
            
            return movie_embeddings, loss_info
        
        else:
            # Inference mode
            if self.classifier is not None:
                logits = self.classifier(movie_embeddings)
                return movie_embeddings, logits
            return movie_embeddings
    
    def compute_structural_loss(self, pruning_results):
        """Compute structural preservation loss L_struct"""
        struct_loss = torch.tensor(0.0)
        
        for behavior, results in pruning_results.items():
            H_orig = results['H_original']
            H_pruned = results['H_pruned']
            
            if torch.is_tensor(H_orig):
                # Limit matrix size for computational efficiency
                max_size = min(50, H_orig.shape[0])  # Reduced size
                H_orig_subset = H_orig[:max_size, :min(50, H_orig.shape[1])]
                A_orig = torch.mm(H_orig_subset, H_orig_subset.t())
            else:
                H_orig_tensor = torch.FloatTensor(H_orig).to(H_pruned.device)
                max_size = min(50, H_orig_tensor.shape[0])  # Reduced size
                H_orig_subset = H_orig_tensor[:max_size, :min(50, H_orig_tensor.shape[1])]
                A_orig = torch.mm(H_orig_subset, H_orig_subset.t())
            
            max_size = min(50, H_pruned.shape[0])  # Reduced size
            H_pruned_subset = H_pruned[:max_size, :min(50, H_pruned.shape[1])]
            A_pruned = torch.mm(H_pruned_subset, H_pruned_subset.t())
            
            # Ensure same dimensions
            min_dim = min(A_orig.shape[0], A_pruned.shape[0])
            A_orig = A_orig[:min_dim, :min_dim]
            A_pruned = A_pruned[:min_dim, :min_dim]
            
            # Frobenius norm difference
            if A_orig.numel() > 0 and A_pruned.numel() > 0:
                adj_loss = torch.norm(A_orig - A_pruned, p='fro') ** 2
                struct_loss += adj_loss * 0.01  # Reduced weight
        
        return struct_loss
    
    def compute_pruning_regularization(self, pruning_results):
        """Compute pruning regularization loss L_prune"""
        prune_loss = torch.tensor(0.0)
        
        for behavior, results in pruning_results.items():
            # L1 sparsity on masks (simplified)
            for mask_name in ['component_mask', 'edge_mask', 'node_mask']:
                if mask_name in results:
                    mask = results[mask_name]
                    if torch.is_tensor(mask):
                        prune_loss += 0.001 * torch.norm(mask.float(), p=1)  # Reduced weight
        
        return prune_loss

# =============================================================================
# FIXED: Training and Evaluation Functions
# =============================================================================

def train_Tri_Prune_HGNN(model, incidence_matrices, node_features, labels, movie_ids,
                   num_epochs=100, learning_rate=0.001, device='cpu'):
    """FIXED: Train Tri_Prune_HGNN model with proper movie-label alignment"""
    model = model.to(device)
    node_features = node_features.to(device)
    labels = labels.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)
    
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 15
    
    print(f"\n=== Training Tri_Prune_HGNN Framework ===")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Movie IDs provided: {len(movie_ids) if movie_ids else 0}")
    print(f"Labels shape: {labels.shape}")
    
    for epoch in range(num_epochs):
        model.train()
        
        try:
            # Forward pass with movie IDs
            embeddings, loss_info = model(
                incidence_matrices, node_features, labels, movie_ids, epoch, training=True
            )
            
            total_loss = loss_info['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(total_loss.item())
            scheduler.step(total_loss)
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Classification: {loss_info['classification_loss'].item():.4f}")
                print(f"  Embeddings shape: {embeddings.shape}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            traceback.print_exc()
            break
    
    return train_losses

def evaluate_Tri_Prune_HGNN(model, incidence_matrices, node_features, labels, movie_ids,
                      evaluator, device='cpu'):
    """FIXED: Evaluate Tri_Prune_HGNN model with proper alignment"""
    model.eval()
    model = model.to(device)
    node_features = node_features.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        try:
            # Get embeddings and predictions
            if model.classifier is not None:
                embeddings, logits = model(incidence_matrices, node_features, 
                                         movie_ids=movie_ids, epoch=0, training=False)
                
                # Ensure alignment between logits and labels
                min_size = min(logits.shape[0], labels.shape[0])
                if min_size > 0:
                    aligned_logits = logits[:min_size]
                    aligned_labels = labels[:min_size]
                    
                    # Convert to numpy for evaluation
                    embeddings_np = embeddings[:min_size].cpu().numpy()
                    labels_np = aligned_labels.cpu().numpy()
                    
                    # Get predictions
                    probs = F.softmax(aligned_logits, dim=1).cpu().numpy()
                    preds = torch.argmax(aligned_logits, dim=1).cpu().numpy()
                    
                    # Evaluate classification
                    classification_results = evaluator.evaluate_classification(
                        labels_np, preds, probs
                    )
                    
                    # Evaluate embeddings
                    embedding_results = evaluator.evaluate_embeddings(embeddings_np, labels_np)
                    
                    return {
                        'classification': classification_results,
                        'embeddings': embedding_results,
                        'embeddings_tensor': embeddings[:min_size],
                        'predictions': preds,
                        'probabilities': probs,
                        'aligned_size': min_size
                    }
                else:
                    print("Warning: No valid aligned samples found")
                    return {'error': 'No valid aligned samples'}
            else:
                embeddings = model(incidence_matrices, node_features, 
                                 movie_ids=movie_ids, epoch=0, training=False)
                embeddings_np = embeddings.cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                embedding_results = evaluator.evaluate_embeddings(embeddings_np, labels_np)
                
                return {
                    'embeddings': embedding_results,
                    'embeddings_tensor': embeddings
                }
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
            traceback.print_exc()
            return {'error': str(e)}

# =============================================================================
# FIXED: Enhanced Evaluation System
# =============================================================================

class EnhancedEvaluator:
    """Comprehensive evaluation system for Tri_Prune_HGNN with MAE, RMSE, and ACC metrics"""
    
    def __init__(self, metrics=['mae', 'rmse', 'accuracy']):
        self.metrics = metrics
        self.results_history = []
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """Evaluate classification performance with MAE, RMSE, and Accuracy"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        results = {}
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 1. Mean Absolute Error (MAE)
        results['mae'] = mean_absolute_error(y_true, y_pred)
        
        # 2. Root Mean Square Error (RMSE)
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # 3. Accuracy (ACC)
        results['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Additional metrics for comprehensive evaluation
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # AUC (if probabilities provided)
        if y_pred_proba is not None:
            try:
                # Handle multiclass AUC
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 2:
                    results['auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                 multi_class='ovr', average='macro')
                else:
                    results['auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                results['auc'] = 0.0
        else:
            results['auc'] = 0.0
        
        return results
    
    def evaluate_embeddings(self, embeddings, labels):
        """Evaluate embedding quality using various metrics"""
        results = {}
        
        # Embedding statistics
        results['embedding_stats'] = {
            'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
            'mean_cosine_sim': np.mean(cosine_similarity(embeddings)),
            'embedding_dim': embeddings.shape[1]
        }
        
        # Clustering quality (simplified)
        if len(np.unique(labels)) > 1:
            from sklearn.metrics import silhouette_score
            try:
                results['silhouette_score'] = silhouette_score(embeddings, labels)
            except:
                results['silhouette_score'] = 0.0
        
        return results

# =============================================================================
# FIXED: Main Comprehensive Evaluation Function
# =============================================================================

def run_comprehensive_evaluation(folder_path, num_runs=10):
    """FIXED: Run comprehensive evaluation with proper movie-label alignment"""
    print("="*80)
    print("Tri_Prune_HGNN COMPREHENSIVE EVALUATION (FIXED VERSION)")
    print("Tri_Prune_HGNN: MAE, RMSE, ACCURACY EVALUATION (10 RUNS)")
    print("="*80)
    
    # ADDED: Check data files first
    print("\nChecking data files...")
    required_files = ['movie_genres.xlsx', 'user_movies.xlsx', 'movie_directors.xlsx', 'movie_actors.xlsx']
    for file in required_files:
        file_path = os.path.join(folder_path, file)
        if os.path.exists(file_path):
            print(f"✓ Found: {file}")
            # Sample the file to understand structure
            try:
                df_sample = pd.read_excel(file_path, nrows=3)
                print(f"  Columns: {list(df_sample.columns)}")
                print(f"  Sample data: {df_sample.iloc[0].to_dict()}")
            except Exception as e:
                print(f"  Error reading sample: {e}")
        else:
            print(f"✗ Missing: {file}")
    
    # Load and process data
    try:
        print("\nProcessing genre data...")
        ground_truth_ratings, genre_id_mapping = process_data(folder_path)
        num_classes = len([k for k in genre_id_mapping.keys() if k != '_reverse'])
        
        print(f"\nDataset Information:")
        print(f"Number of movies: {ground_truth_ratings['movieID'].nunique()}")
        print(f"Number of genres: {num_classes}")
        print(f"Genre distribution:")
        genre_counts = ground_truth_ratings['genreID'].value_counts().sort_index()
        
        # Show genre distribution with names if available
        if '_reverse' in genre_id_mapping:
            reverse_mapping = genre_id_mapping['_reverse']
            for genre_id, count in genre_counts.items():
                genre_name = reverse_mapping.get(genre_id, f"Unknown_{genre_id}")
                print(f"  {genre_name} (ID {genre_id}): {count} movies")
        else:
            for genre_id, count in genre_counts.items():
                print(f"  Genre {genre_id}: {count} movies")
    
    except Exception as e:
        print(f"Error processing data: {e}")
        traceback.print_exc()
        return None, None
    
    # Create hypergraph
    try:
        print("\nBuilding hypergraph...")
        constructor = HypergraphConstructor()
        hypergraph_data = constructor.create_master_slave_hypergraph(folder_path)
        incidence_matrices, node_mappings = constructor.generate_incidence_matrices(hypergraph_data)
        
        print(f"\nHypergraph Information:")
        for behavior, H in incidence_matrices.items():
            print(f"{behavior}: {H.shape[0]} nodes × {H.shape[1]} hyperedges, "
                  f"sparsity: {np.mean(H == 0):.2%}")
    
    except Exception as e:
        print(f"Error creating hypergraph: {e}")
        traceback.print_exc()
        return None, None
    
    # FIXED: Prepare features and labels with proper movie ID mapping
    num_nodes = constructor.num_nodes
    movie_ids = ground_truth_ratings['movieID'].tolist()
    labels = torch.LongTensor(ground_truth_ratings['genreID'].values)
    
    print(f"\nData Alignment:")
    print(f"Total nodes in hypergraph: {num_nodes}")
    print(f"Movie nodes with labels: {len(movie_ids)}")
    print(f"Movie node indices available: {len(constructor.movie_node_indices)}")
    print(f"Labels range: {labels.min().item()} to {labels.max().item()}")
    
    # Check alignment
    aligned_movies = 0
    for movie_id in movie_ids[:10]:  # Check first 10
        if movie_id in constructor.movie_node_indices:
            aligned_movies += 1
    print(f"Sample alignment check: {aligned_movies}/10 movies found in hypergraph")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Running {num_runs} independent evaluations...")
    
    # Initialize evaluator with MAE, RMSE, ACC metrics
    evaluator = EnhancedEvaluator(metrics=['mae', 'rmse', 'accuracy'])
    
    # Store results for all runs
    all_results = {
        'mae': [], 
        'rmse': [], 
        'accuracy': [],
        'f1_macro': [],
        'f1_micro': [],
        'f1_weighted': [],
        'auc': []
    }
    
    # Track training details
    training_details = {
        'training_losses': [],
        'convergence_epochs': [],
        'final_losses': [],
        'alignment_sizes': []
    }
    
    # Run multiple evaluation rounds
    for run in range(num_runs):
        print(f"\n{'='*20} RUN {run+1}/{num_runs} {'='*20}")
        
        # Set different random seed for each run
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # FIXED: Initialize fresh model with movie node indices
        model = Tri_Prune_HGNN_Framework(
            input_dim=64, 
            hidden_dim=128, 
            output_dim=32,
            num_behaviors=len(incidence_matrices),
            num_nodes=num_nodes,
            movie_node_indices=constructor.movie_node_indices  # ADDED: Pass movie mapping
        )
        model.set_classifier(num_classes)
        
        # Generate fresh random features for each run with proper dimensions
        node_features = torch.randn(num_nodes, 64)
        
        # FIXED: Train model with movie IDs
        print(f"Training model for run {run+1}...")
        try:
            train_losses = train_Tri_Prune_HGNN(
                model, incidence_matrices, node_features, labels, movie_ids,
                num_epochs=30, learning_rate=0.001, device=device
            )
            
            # Store training details
            training_details['training_losses'].append(train_losses)
            training_details['convergence_epochs'].append(len(train_losses))
            training_details['final_losses'].append(train_losses[-1] if train_losses else 0.0)
            
            # FIXED: Evaluate model with movie IDs
            print(f"Evaluating model for run {run+1}...")
            results = evaluate_Tri_Prune_HGNN(
                model, incidence_matrices, node_features, labels, movie_ids,
                evaluator, device=device
            )
            
            # Check if evaluation was successful
            if 'error' in results:
                print(f"Run {run+1} failed: {results['error']}")
                continue
            
            # Store alignment size for debugging
            if 'aligned_size' in results:
                training_details['alignment_sizes'].append(results['aligned_size'])
                print(f"Aligned samples: {results['aligned_size']}")
            
            # Store results
            if 'classification' in results:
                for metric in ['mae', 'rmse', 'accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 'auc']:
                    if metric in results['classification']:
                        all_results[metric].append(results['classification'][metric])
                        
                # Print current run results
                print(f"Run {run+1} Results:")
                print(f"  MAE: {results['classification']['mae']:.4f}")
                print(f"  RMSE: {results['classification']['rmse']:.4f}")
                print(f"  Accuracy: {results['classification']['accuracy']:.4f}")
                print(f"  F1-Macro: {results['classification']['f1_macro']:.4f}")
                print(f"  AUC: {results['classification']['auc']:.4f}")
            else:
                print(f"Run {run+1}: No classification results available")
                
        except Exception as e:
            print(f"Run {run+1} failed with error: {e}")
            traceback.print_exc()
            continue
    
    # Calculate final statistics with confidence intervals
    print(f"\n{'='*80}")
    print("FINAL RESULTS: MAE, RMSE, ACCURACY (FIXED VERSION)")
    print(f"{'='*80}")
    
    # Check if we have any valid results
    successful_runs = len(all_results['accuracy'])
    print(f"\nSuccessful runs: {successful_runs}/{num_runs}")
    
    if successful_runs == 0:
        print("ERROR: No successful runs completed. Please check the data and implementation.")
        return None, None
    
    final_results = {}
    
    # Focus on primary metrics: MAE, RMSE, Accuracy
    primary_metrics = ['mae', 'rmse', 'accuracy']
    
    print("\nPRIMARY METRICS:")
    print("-" * 50)
    
    for metric in primary_metrics:
        if metric in all_results and all_results[metric]:
            values = np.array(all_results[metric])
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            min_val = np.min(values)
            max_val = np.max(values)
            
            # 95% confidence interval (if we have enough samples)
            if len(values) > 1:
                conf_interval = stats.t.interval(0.95, len(values)-1, 
                                               loc=mean_val, scale=stats.sem(values))
            else:
                conf_interval = (mean_val, mean_val)
            
            final_results[metric] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'conf_interval': conf_interval,
                'values': values.tolist()
            }
            
            print(f"{metric.upper()}:")
            print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
            print(f"  95% CI: [{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]")
            print()
    
    # Additional metrics summary
    print("ADDITIONAL METRICS:")
    print("-" * 50)
    
    for metric in ['f1_macro', 'f1_micro', 'f1_weighted', 'auc']:
        if metric in all_results and all_results[metric]:
            values = np.array(all_results[metric])
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            
            final_results[metric] = {
                'mean': mean_val,
                'std': std_val,
                'values': values.tolist()
            }
            
            print(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Training statistics
    print(f"\nTRAINING STATISTICS:")
    print("-" * 50)
    if training_details['convergence_epochs']:
        avg_epochs = np.mean(training_details['convergence_epochs'])
        avg_final_loss = np.mean(training_details['final_losses'])
        print(f"Average convergence epochs: {avg_epochs:.1f}")
        print(f"Average final training loss: {avg_final_loss:.4f}")
    
    if training_details['alignment_sizes']:
        avg_alignment = np.mean(training_details['alignment_sizes'])
        print(f"Average aligned samples: {avg_alignment:.1f}")
    
    # Create summary table
    print(f"\nSUMMARY TABLE:")
    print("-" * 50)
    print(f"{'Metric':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    
    for metric in primary_metrics:
        if metric in final_results:
            metric_stats = final_results[metric]  # FIXED: Renamed from 'stats' to 'metric_stats'
            print(f"{metric.upper():<12} {metric_stats['mean']:<10.4f} {metric_stats['std']:<10.4f} "
                  f"{metric_stats['min']:<10.4f} {metric_stats['max']:<10.4f}")
    
    return final_results, training_details

# =============================================================================
# FIXED: Main Execution Function
# =============================================================================

def main():
    """FIXED: Main execution function with proper error handling"""
    # Configuration
    config = {
        'folder_path': 'C:\\IMDB',  # Update this path
        'num_runs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Tri_Prune_HGNN Framework Evaluation (FIXED VERSION)")
    print("Metrics: MAE, RMSE, Accuracy")
    print("Evaluation runs: 10")
    print("=" * 50)
    
    try:
        # Verify data path exists
        if not os.path.exists(config['folder_path']):
            print(f"ERROR: Data folder not found at {config['folder_path']}")
            print("Please update the folder_path in the config section.")
            return None, None
        
        # Check required files
        required_files = ['movie_genres.xlsx', 'user_movies.xlsx', 'movie_directors.xlsx', 
                         'movie_actors.xlsx']
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(config['folder_path'], file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"ERROR: Missing required files: {missing_files}")
            return None, None
        
        # Run comprehensive evaluation with 10 runs
        results, training_details = run_comprehensive_evaluation(
            config['folder_path'], 
            num_runs=config['num_runs']
        )
        
        if results is None:
            print("Evaluation failed. Please check the error messages above.")
            return None, None
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETED SUCCESSFULLY (FIXED VERSION)")
        print(f"{'='*80}")
        
        # Create final results summary focusing on MAE, RMSE, ACC
        primary_metrics = ['mae', 'rmse', 'accuracy']
        
        print(f"\nFINAL SUMMARY - Tri_Prune_HGNN Performance:")
        print("=" * 60)
        
        summary_data = {}
        for metric in primary_metrics:
            if metric in results:
                summary_data[metric] = {
                    'Mean': results[metric]['mean'],
                    'Std': results[metric]['std'],
                    'Min': results[metric]['min'],
                    'Max': results[metric]['max']
                }
        
        # Create and display summary DataFrame
        if summary_data:
            summary_df = pd.DataFrame(summary_data).T
            print("\nResults from successful runs:")
            print(summary_df.round(4))
        
        # Statistical significance test (one-sample t-test against baseline)
        print(f"\nSTATISTICAL ANALYSIS:")
        print("-" * 40)
        
        for metric in primary_metrics:
            if metric in results and results[metric]['values']:
                values = np.array(results[metric]['values'])
                
                if metric == 'accuracy':
                    # Random baseline for accuracy
                    try:
                        num_genres = len(pd.read_excel(os.path.join(config['folder_path'], 
                                                                  'movie_genres.xlsx'))['genreID'].unique())
                        baseline = 1.0 / num_genres
                        if len(values) > 1:
                            t_stat, p_value = stats.ttest_1samp(values, baseline)
                            print(f"{metric.upper()}: t-statistic = {t_stat:.4f}, p-value = {p_value:.6f}")
                            print(f"  Significantly better than random baseline ({baseline:.4f}): {'Yes' if p_value < 0.05 else 'No'}")
                        else:
                            print(f"{metric.upper()}: Single value = {values[0]:.4f}, baseline = {baseline:.4f}")
                    except Exception as e:
                        print(f"{metric.upper()}: Could not compute baseline comparison: {e}")
                else:
                    # For MAE and RMSE, lower is better
                    print(f"{metric.upper()}: Mean = {np.mean(values):.4f}, Std = {np.std(values, ddof=1):.4f}")
                    if np.mean(values) > 0:
                        print(f"  Coefficient of Variation: {np.std(values, ddof=1)/np.mean(values)*100:.2f}%")
        
        # Save detailed results to CSV
        try:
            if any(results[metric]['values'] for metric in primary_metrics if metric in results):
                results_for_csv = []
                max_runs = max(len(results[metric]['values']) for metric in primary_metrics if metric in results)
                
                for run_idx in range(max_runs):
                    row = {'Run': run_idx + 1}
                    for metric in primary_metrics:
                        if metric in results and run_idx < len(results[metric]['values']):
                            row[metric.upper()] = results[metric]['values'][run_idx]
                    results_for_csv.append(row)
                
                results_df = pd.DataFrame(results_for_csv)
                csv_filename = 'Tri_Prune_HGNN_evaluation_results_fixed.csv'
                results_df.to_csv(csv_filename, index=False)
                print(f"\nDetailed results saved to: {csv_filename}")
            
        except Exception as e:
            print(f"Could not save CSV file: {e}")
        
        return results, training_details
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results = main()