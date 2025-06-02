# Quick Start
    # Installation of dependencies
    pip install torch torch-geometric scikit-learn numpy torch_cluster
    python rwgnn_pyg.py --dataset IMDB-BINARY


1. Overview
   RWNN implements a graph classification model based on random walks.

   * On the input graph, node features x are repeatedly updated via sparse multiplication (x ← A·x) to simulate random walks.
   * In parallel, a set of small “hidden graphs” with learnable adjacency and node features performs its own graph convolutions.
   * At each step the main‐graph and hidden‐graph features are fused, then aggregated to produce a graph‐level representation.
   * A two‐layer MLP classifies each graph into target classes.

2. Directory Structure
   main.py            – entry point: argument parsing, 10‐fold CV, train/test loops
   model.py           – PyTorch RWNN model (class RW\_NN)
   rwgnn\_pyg.py       – optional PyG MessagePassing version (class RW\_GNN)
   utils.py           – data I/O, batch construction, accuracy and meter utilities
   readme.txt         – this file
   datasets/          – TU‐format datasets

3. Requirements

   * Python 3.8+
   * PyTorch
   * SciPy
   * scikit‐learn
   * tqdm
   * [optional] torch_sparse, torch_geometric

   Install example:
   pip install torch torchvision scipy scikit-learn tqdm
   pip install torch_sparse torch_geometric

4. Data Preparation

   1. Place TU‐format files under datasets
   2. main.py uses utils.load_data() to read and split them into per‐graph adjacency and feature lists.



   Key options:
   --dataset            Dataset folder name
   --use-node-labels    Use node label one-hot features
   --lr                 Learning rate
   --max-step           Maximum random-walk steps L
   --hidden-graphs      Number of hidden graphs H
   --size-hidden-graphs Nodes per hidden graph S
   --hidden-dim         Hidden feature dimension D
   --penultimate-dim    MLP penultimate layer size P
   --batch-size         Number of graphs per batch
   --epochs             Number of training epochs
   --normalize          Divide by node counts to average during aggregation

   The script performs 10-fold cross-validation, prints train/val logs per fold, and reports final mean test accuracy.

6. Code Flow

   utils.load_data(): Read TU‐format files, return lists of SciPy adj matrices, feature arrays, and graph labels.
   utils.generate_batches(): Pack multiple small graphs into one large sparse adj and feature tensor, create graph_indicator and label tensors per batch.
   model.RW_NN:
     • Learnable hidden‐graph adjacencies and node features
     • Map input features → D dims, sigmoid activation
     • For i in 0…L-1:
         Main‐graph: x ← A·x
         Hidden‐graph: z ← A_hidden·z
         Fuse: elementwise multiply main and hidden features
         Aggregate by graph_indicator → [n_graphs×H]
     • Concatenate L steps → [n_graphs×(H·L)]
     • BatchNorm → fc1+ReLU+dropout → fc2 → log_softmax
   rwgnn_pyg.RW_GNN: Same logic using PyG MessagePassing for main‐graph convolution.
   main.py: Parses arguments, loads data, splits 10 folds, builds batches, initializes model/optimizer/scheduler, runs train() and test() per epoch, saves best model by validation accuracy, evaluates on test set, and averages results across folds.
