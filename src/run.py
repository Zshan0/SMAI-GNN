"""
1. parse all the arguments
2. get the graphs object
3. get the cross validation sets
4. Call the model class
5. iterate over the sets to train
"""
import json
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.gnn import GNN

criterion = nn.CrossEntropyLoss()

from tqdm import tqdm
from data_handling import parse_dataset, k_fold_splitter


def parse_arguments():
    parser = argparse.ArgumentParser(description="SMAI project team 30")
    parser.add_argument(
        "--dataset",
        type=str,
        default="PROTEINS",
        help="name of dataset (default: PROTEINS)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=350,
        help="number of epochs to train (default: 350)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for splitting the dataset into 10 (default: 0)",
    )
    parser.add_argument(
        "--fold_idx",
        type=int,
        default=10,
        help="The idx of fold to train on",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=5,
        help="number of layers INCLUDING the input one (default: 5)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="number of hidden units (default: 64)",
    )
    parser.add_argument(
        "--degree_as_tag",
        action="store_true",
        help="let the input node features be the degree of nodes (heuristics for unlabeled graph)",
    )
    parser.add_argument(
        "--name",
        default=f"GNN-{int(time.time())}",
        help="Save file name for the model",
    )
    parser.add_argument(
        "--output_folder",
        default=f".",
        help="Where the output file is stored",
    )
    return parser.parse_args()


def train(args, model, train_graphs, optimizer, epoch):
    model.train()

    print("Epoch", epoch)
    losses = []
    for _ in tqdm(range(50), unit="batch"):
        selected_idx = np.random.permutation(len(train_graphs))[
                       : args.batch_size
                       ]

        batch_graph = [train_graphs[idx] for idx in selected_idx]

        labels = torch.LongTensor([graph.label for graph in batch_graph])
        loss = criterion(model(batch_graph), labels)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        losses.append(loss)

    return np.average(np.array(losses))


def get_accuracy(model, graphs):
    pred = model(graphs).detach()
    pred = pred.argmax(1, keepdim=True)
    labels = torch.LongTensor([graph.label for graph in graphs])
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(graphs))
    return acc_train


def test(model, train_graphs, test_graphs):
    model.eval()
    return get_accuracy(model, train_graphs), get_accuracy(model, test_graphs)


def main():
    args = parse_arguments()
    torch.manual_seed(0)
    np.random.seed(0)

    graphs, num_classes = parse_dataset(args.dataset, args.degree_as_tag)

    train_set, test_set = k_fold_splitter(graphs, args.seed, args.fold_idx)
    print(num_classes)

    model = GNN(
        train_set[0].node_features.shape[1],
        args.num_layers,
        args.hidden_dim,
        num_classes,
        False,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    accuracies = []
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        avg_loss = train(args, model, train_set, optimizer, epoch)
        acc = test(
            model, train_set, test_set
        )
        accuracies.append((epoch, float(avg_loss), float(acc[0]), float(acc[1])))
        print(avg_loss, acc[0], acc[1])

    with open(f"{args.output_folder}/result-{args.dataset}-{args.fold_idx}-{args.seed}-{args.name}.json", "w") as f:
        json.dump(accuracies, f)


if __name__ == "__main__":
    main()
