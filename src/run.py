"""
1. parse all the arguments
2. get the graphs object
3. get the cross validation sets
4. Call the model class
5. iterate over the sets to train
"""
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.gnn import GNN

criterion = nn.CrossEntropyLoss()

from tqdm import tqdm
from data_handling import parse_dataset, k_fold_splitter


def pass_data_iteratively(model, graphs, minibatch_size=64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i: i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


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
    return parser.parse_args()


def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = 50
    pbar = tqdm(range(total_iters), unit="batch")

    loss_accum = 0
    for _ in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[
                       : args.batch_size
                       ]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(
            device
        )

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description("epoch: %d" % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


def test(model, device, train_graphs, test_graphs):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(
        device
    )
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test


def main():
    args = parse_arguments()
    torch.manual_seed(0)
    np.random.seed(0)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = parse_dataset(args.dataset, args.degree_as_tag)

    train_set, test_set = k_fold_splitter(graphs, args.seed, args.fold_idx)
    print(num_classes)

    model = GNN(
        train_set[0].node_features.shape[1],
        args.num_layers,
        args.hidden_dim,
        num_classes,
        False,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    accuracies = []
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        avg_loss = train(args, model, device, train_set, optimizer, epoch)
        acc_train, acc_test = test(
            model, device, train_set, test_set
        )
        accuracies.append((epoch, avg_loss, acc_train, acc_test))
        print(avg_loss, acc_train, acc_test)

    with open(f"result-{args.dataset}-{args.fold_idx}-{args.seed}.json", "w") as f:
        json.dump(accuracies, f)


if __name__ == "__main__":
    main()
