# Quick Start
    # Installation of dependencies
    pip install torch torch-geometric scikit-learn numpy torch_cluster
    python rwgnn_pyg_stop.py --dataset IMDB-BINARY


1. Overview
a. Sample several random walk sequences from the input graph (such as Cora, IMDB-BINARY) to obtain a set of "node ID sequences".
b. Map each walk sequence into a vector representation (RNN, 1D-CNN or Transformer can be used) to obtain the walk features of the graph.
c. Meanwhile, a set of learnable "hidden graphs" is introduced. Graph convolution is performed on these hidden small graphs to extract high-order structural features.
d. Fuse and aggregate the "walk representation of real images" and the "graph convolution representation of hidden images" to the graph level, and then send them to MLP for classification.

The approach involves sampling multiple discrete random walk paths from a graph structure,
then encoding each path using a Recurrent Neural Network (RNN).
The features of all paths in a single graph are aggregated into a global vector representation, which is subsequently used for downstream classification tasks.

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
   1.Data Preparation

1. Use datasets from TUDataset (provided by PyG).
2. If original nodes lack explicit features (i.e., data.x is None), construct initial features using one-hot encoding of node degrees, enabling the degree information to serve as features.

2.Batch Construction (via PyG DataLoader)

The PyG DataLoader combines multiple independent graphs in a batch into a “super graph” (block-diagonal adjacency and feature matrices). A batch tensor is generated to indicate which subgraph each node belongs to.

For example, with a batch size of 4, the DataLoader stacks all nodes from four graphs, resulting in a data.batch tensor whose length equals the total number of nodes in the batch, with values ranging from 0 to 3, indicating subgraph membership.

This design allows multiple graphs to be processed in parallel during a single forward pass, avoiding manual stitching logic for processing each graph individually.

3.Explicit Random Walk

Core logic of Random Walk:

a. For each subgraph, identify all its node indices (via batch == graph_idx), then extract edge lists for these nodes from edge_index and convert to an undirected representation.

b. Randomly select up to num_walks starting nodes (if fewer nodes are available, use all of them), and invoke torch_cluster.random_walk(row, col, start, walk_length=L-1).

c. This API returns a node index sequence of length walk_length for each start node, with walk behavior halted at boundaries or isolated nodes (remaining stationary).

d. If a sequence is shorter than walk_length due to dead ends, it is padded with the last node; if longer, it is truncated.

e. The result is a 2D tensor:

walks.shape = [total_walks, walk_length]
total_walks = (number of graphs) × (number of walks per graph)

Additionally, a 1D tensor walk_batch records the graph index for each walk:
walk_batch.shape = [total_walks]


4.Path Sequence Feature Encoding (GRU)

Each original node has a feature vector x[node_idx] ∈ ℝ^F. A linear layer feature_encoder = Linear(F, D) projects it into a hidden space ℝ^D.

For walk i, with indices walks[i], extract encoded features x_encoded[node_idx] to form:

walk_features.shape = [total_walks, walk_length, hidden_dim]

Feed walk_features into a GRU (hidden_dim → hidden_dim, batch_first=True) and retain only the last hidden state:

path_encodings.shape = [total_walks, hidden_dim]

This condenses each walk into a D-dimensional vector.

5.Graph-Level Feature Aggregation

Use walk_batch to retrieve walk features belonging to each graph and compute the mean:

graph_encodings[k] = path_encodings[mask].mean(dim=0)

Final output:
graph_encodings.shape = [batch_size, hidden_dim]

6.BatchNorm + MLP Classification

a. Apply BatchNorm1d(hidden_dim) on graph_encodings to normalize features across the batch.
b. Pass through fully connected layers → ReLU → Dropout → final FC → log_softmax:

out = self.relu(self.fc1(graph_encodings))
out = self.dropout(out)
out = self.fc2(out)
return F.log_softmax(out, dim=1)

7.Training/Validation/Test Pipeline

a. Use PyG DataLoader to build loaders for training/validation/test sets.

train(model, loader, optimizer, device):

- Set model to train mode
- Iterate over batches, call model(data), compute loss with F.cross_entropy(out, data.y), backpropagate and update
- Track total loss and correct predictions

b. test(model, loader, device):

- Set model to eval mode and use no_grad
- Same forward pass, no backward

c. Each epoch:

- Run train() on training set
- Run test() on validation set
- Adjust LR using scheduler
- Apply early stopping based on validation loss

d. After each fold, load best model from checkpoint_path and evaluate on test set.

e. After 10 folds, report mean and std of test accuracy.

8.Visualization and Results

a. During the training process, information such as the current epoch, train_loss, train_acc, val_loss, val_acc, and elapsed time will be printed in real time.
b. When the validation set loss is less than the previous best value, EarlyStopping will save the current parameters and inform in the print log that "Validation loss decreased, model saved".
c. In the testing phase, it will print "Test result - Loss: XX, Accuracy: XX", and finally output the "average test accuracy ± standard deviation" of 10 folds as well as the "list of accuracies for each fold".

   main.py: Parses arguments, loads data, splits 10 folds, builds batches, initializes model/optimizer/scheduler, runs train() and test() per epoch, saves best model by validation accuracy, evaluates on test set, and averages results across folds.
