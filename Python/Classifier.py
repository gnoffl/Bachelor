from typing import List

import sklearn.tree as tree
import graphviz
import os
import pickle
from abc import ABC, abstractmethod

import dataCreation as dc
import visualization as vs


class Classifier(ABC):
    @abstractmethod
    def predict(self, dataset: dc.Data):
        pass

    @staticmethod
    def load_classifier(dataset: dc.Data):
        tree_path = os.path.join(dataset.path, "tree_classifier.pkl")
        if os.path.isfile(tree_path):
            with open(tree_path, "rb") as f:
                decision_tree = pickle.load(f)
            return TreeClassifier(trained_tree=decision_tree)
        else:
            raise dc.CustomError("No Tree was saved, so no tree can be loaded!")


class TreeClassifier(Classifier):
    model: tree.DecisionTreeClassifier

    def __init__(self, dataset: dc.Data = None, depth: int = 5, min_samples_leaf: int = 5,
                 trained_tree: tree.DecisionTreeClassifier = None):
        """
        trains a decision tree on a DataSet, uses the tree to predict the classes and saves
        the resulting dataset. Also saves the resulting tree at the path of the dataset.
        :param dataset: the data on which predictions will be made
        :param depth: maximal depth of the decision tree
        :param min_samples_leaf: minimum number of data points per leaf in the decision tree
        """
        if trained_tree is not None:
            self.model = trained_tree
        else:
            if dataset is None:
                raise dc.CustomError("invalid combination of parameters. Either dataset or trained_tree need to be"
                                     "given!")
            self.train_decision_tree(dataset, max_depth=depth, min_samples_leaf=min_samples_leaf)

            dataset.extend_notes_by_one_line(f"Trained DecisionTree!")
            dataset.extend_notes_by_one_line(f"Parameters: max_depth={depth}, min_samples_leaf={min_samples_leaf}")
            tree_path = os.path.join(dataset.path, "tree_classifier.pkl")
            if not os.path.isfile(tree_path):
                with open(tree_path, "wb") as f:
                    pickle.dump(self.model, f)

            dataset.end_paragraph_in_notes()

    def visualize_predictions(self, dataset: dc.Data, pred_col_name: str) -> str:
        self.predict_classes(dataset=dataset, pred_col_name=pred_col_name)
        dataset.extend_notes_by_one_line(f"Predicted classes using Decision Tree in column \"{pred_col_name}\".")
        tree_pics_path = vs.visualize_model_predictions(dataset=dataset, pred_col_name=pred_col_name)

        return tree_pics_path

    @staticmethod
    def load_tree(tree_path: str) -> tree.DecisionTreeClassifier:
        """
        loads a decisionTree from a given Path
        :param tree_path: path to the pickled tree
        :return: the loaded tree object
        """
        with open(tree_path, "rb") as f:
            print("loading tree!")
            decision_tree = pickle.load(f)
        return decision_tree

    def train_decision_tree(self, dataset: dc.Data, max_depth=5, min_samples_leaf=5) -> None:
        """
        trains a decision tree for the given data
        :param dataset: data
        :param max_depth: max depth of the decision tree
        :param min_samples_leaf: minimum samples per leaf in the decision tree
        :return: the trained tree
        """
        df = dataset.data
        tree_path = os.path.join(dataset.path, "tree_classifier.pkl")
        if not os.path.isfile(tree_path):
            classes = df["classes"]
            values = df[dataset.data_columns]
            self.model = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            self.model.fit(values, classes)
        else:
            self.model = self.load_tree(tree_path=tree_path)

    def predict_classes(self, dataset: dc.Data, pred_col_name: str) -> None:
        """
        uses a trained decision tree to predict the classes of the given data
        :param trained_model: the trained decision tree
        :param dataset: the data
        :param pred_col_name: name for the column, in which the predicted classes will be stored
        """
        data = dataset.data
        if pred_col_name in data.columns:
            raise dc.CustomError("Column already exists!")
        values = data[dataset.data_columns]
        results = self.model.predict(values)
        data[pred_col_name] = results

    def visualize_tree(self, dataset: dc.Data, tree_pics_path: str) -> None:
        """
        creates a pdf file of the given decision tree in the folder of the dataset
        :param dataset: dataset, for which the decision tree is relevant. Attribute "path" will be used as the location for
        the resulting pdf file
        :param trained_tree: the decision tree that is to be visualized
        :param tree_pics_path: Path were pictures for the visualization of the tree is supposed to be saved
        """
        df = dataset.data
        values = df[dataset.data_columns]
        visualization_path = os.path.join(dataset.path, "tree_visualization_data.gv")
        with open(visualization_path, "w") as f:
            tree.export_graphviz(self.model, out_file=f,
                                 feature_names=values.columns,
                                 filled=True, rounded=True,
                                 special_characters=True)
        with open(visualization_path, "r") as f2:
            content = f2.read()
            graph = graphviz.Source(content, filename="tree_visualization", directory=tree_pics_path)
            graph.render()
        # clean up unnecessary files
        vis_path = os.path.join(tree_pics_path, "tree_visualization")
        if os.path.isfile(vis_path):
            os.remove(vis_path)
        gv_path = os.path.join(dataset.path, "tree_visualization_data.gv")
        if os.path.isfile(gv_path):
            os.remove(gv_path)

    def predict(self, dataset: dc.Data):
        return self.model.predict(dataset.data)


from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import dataCreation as dc


class Network(nn.Module):
    def __init__(self, input_size: int, nr_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, nr_classes)

    def forward(self, x):
        input1 = self.fc1(x)
        hidden = F.relu(input1)
        output = self.fc2(hidden)
        return output


def get_data_loaders(dataset: dc.Data, batch_size: int = 64, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    training, testing = dataset.get_test_training_split()
    training_loader = DataLoader(dataset=training, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=testing, batch_size=batch_size, shuffle=shuffle)
    return training_loader, test_loader


def check_accuracy(loader: DataLoader, model: Network, device: torch.device):
    num_correct = num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct/num_samples) * 100:.2f}")


def train(dataset: dc.Data, lr: float = 0.001, num_epochs: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network(input_size=len(dataset.data_columns), nr_classes=len(dataset.class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader, test_loader = get_data_loaders(dataset=dataset)

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            #forward
            scores = model(data)
            loss = criterion(scores, targets.long())

            #backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        print("testing training data")
        check_accuracy(loader=train_loader, model=model, device=device)
        print("testing test data")
        check_accuracy(loader=test_loader, model=model, device=device)


def test_data_generation():
    data = dc.SoccerDataSet(save=False)
    training_loader, test_loader = get_data_loaders(data)
    for idx, (data, target) in enumerate(training_loader):
        print(idx)
        print(data.shape)
        print(target.shape)
        print(data[1, :])
        break


def test_nn():
    dataset = dc.SoccerDataSet(save=False)
    train(dataset=dataset, num_epochs=100)
    #model = Network(784, 10)
    #test_ = torch.randn(64, 784)
    #print(model(test_).shape)


def test_tree() -> None:
    """
    runs the training process and visualizes results
    """
    #members = [1000 for _ in range(6)]
    #dataset = dc.MaybeActualDataSet(members)
    #dataset = dc.MaybeActualDataSet.load("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220226_135403_MaybeActualDataSet")
    #dataset = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\220428_124321_IrisDataSet")
    dataset = dc.SoccerDataSet()
    classifier = TreeClassifier(dataset)
    path = classifier.visualize_predictions(dataset=dataset, pred_col_name="test")
    classifier.visualize_tree(dataset=dataset, tree_pics_path=path)
    #print(tree_)
    #dataset.run_hics()
    """dataset = dc.MaybeActualDataSet.load("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220226_135403_MaybeActualDataSet")
    trained_tree = dataset.load_tree()"""
    #matrix = vs.get_change_matrix(data, ("classes", "predicted_classes"))
    #print(matrix)


if __name__ == "__main__":
    test_nn()
    #test_tree()
    #test_data_generation()
