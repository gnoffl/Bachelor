from __future__ import annotations
from typing import List, Tuple

import sklearn.tree as tree
import graphviz
import os
import pickle
from abc import ABC, abstractmethod

import visualization as vs
import dataCreation as dc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Classifier(ABC):
    @abstractmethod
    def predict(self, dataset: dc.Data) -> List:
        """
        use the trained model to predict classes from a set of inputs given in a dataset.
        :param dataset: contains the input data
        :return: predictions in form of a list
        """
        pass

    @staticmethod
    def load_classifier(dataset: dc.Data) -> Classifier:
        """
        loads DectisionTrees. Doenst work for NNs yet.
        :param dataset: dataset, in which the tree is saved
        :return: Classifier object
        """
        tree_path = os.path.join(dataset.path, "tree_classifier.pkl")
        model_path = os.path.join(dataset.path, "model.pkl")
        if os.path.isfile(tree_path):
            with open(tree_path, "rb") as f:
                decision_tree = pickle.load(f)
            return TreeClassifier(trained_tree=decision_tree)
        elif os.path.isfile(model_path):
            return torch.load(model_path)
        else:
            raise dc.CustomError("No Model was saved, so no model can be loaded!")


class TreeClassifier(Classifier):
    model: tree.DecisionTreeClassifier

    def __init__(self, dataset: dc.Data = None, depth: int = 5, min_samples_leaf: int = 5,
                 trained_tree: tree.DecisionTreeClassifier = None):
        """
        trains a decision tree on a DataSet and saves the resulting tree at the path of the dataset.
        :param dataset: the data which will be used for training
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

            #comments for dataset
            dataset.extend_notes_by_one_line(f"Trained DecisionTree!")
            dataset.extend_notes_by_one_line(f"Parameters: max_depth={depth}, min_samples_leaf={min_samples_leaf}")
            dataset.end_paragraph_in_notes()

            #saving model
            tree_path = os.path.join(dataset.path, "tree_classifier.pkl")
            if not os.path.isfile(tree_path):
                with open(tree_path, "wb") as f:
                    pickle.dump(self.model, f)

    def visualize_predictions(self, dataset: dc.Data, pred_col_name: str) -> str:
        """
        visualizes predictions for the model on a given dataset. Predictions will be saved in the pred_col_name.
        Pictures will be saved in the folder of the dataset, under pics/Classifier
        :param dataset: dataset to do the visualization on
        :param pred_col_name: name of the generated column with the predictions
        :return: string of the location of the saved pictures
        """
        self.predict_classes(dataset=dataset, pred_col_name=pred_col_name)
        dataset.extend_notes_by_one_line(f"Predicted classes using Decision Tree in column \"{pred_col_name}\".")
        dataset.end_paragraph_in_notes()
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

    def predict(self, dataset: dc.Data) -> List:
        """
        makes predictions on the given data with the trained DecisionTreeClassifier
        :param dataset: input Data
        :return: predicted classes
        """
        return self.model.predict(dataset.data)


class NNClassifier(nn.Module, Classifier):
    def __init__(self, dataset: dc.Data, lr: float = 0.001, num_epochs: int = 3, batch_size: int = 64,
                 shuffle: bool = True):
        """
        trains a Neural Network on the given DataSet
        :param dataset: input Data
        :param lr: learning rate of the Neural Network training process
        :param num_epochs: number of epochs
        :param batch_size: Batch Size
        :param shuffle: determines whether the content of the batches will be shuffled after the first epoch
        """
        super().__init__()
        torch.manual_seed(42)
        input_size = len(dataset.data_columns)
        nr_classes = len(dataset.class_names)
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, nr_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_the_net(dataset=dataset, lr=lr, num_epochs=num_epochs, batch_size=batch_size, shuffle=shuffle)
        dataset.extend_notes_by_one_line(f"Trained NeuralNetwork!")
        dataset.extend_notes_by_one_line(f"Parameters: lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}, "
                                         f"shuffle={shuffle}")
        dataset.end_paragraph_in_notes()

        model_path = os.path.join(dataset.path, "model.pkl")
        if not os.path.isfile(model_path):
            torch.save(self, model_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through the network
        :param x: input data
        :return: network output
        """
        input1 = self.fc1(x)
        hidden1 = F.relu(input1)
        hidden2 = F.relu(self.fc2(hidden1))
        output = self.fc3(hidden2)
        return output

    @staticmethod
    def get_data_loaders(dataset: dc.Data, batch_size: int = 64, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        creates a dataloader object for the training dataset as well as the test dataset
        :param dataset: source Dataset, that will be split into test and training dataset
        :param batch_size: size of the batches that will be used to train the Neural Network
        :param shuffle: determines whether the content of the batches will be shuffled after the first epoch
        :return: Tuple of the dataloaders (Training, Test)
        """
        training, testing = dataset.get_test_training_split()
        training_loader = DataLoader(dataset=training, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(dataset=testing, batch_size=batch_size, shuffle=shuffle)
        return training_loader, test_loader

    def check_accuracy(self, loader: DataLoader) -> None:
        """
        calculates the accuracy of the Neural Net on the data given by the Dataloader and prints it.
        :param loader: input data
        """
        num_correct = num_samples = 0
        self.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                scores = self(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
            print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct/num_samples) * 100:.2f}")

    def train_the_net(self, dataset: dc.Data, lr: float = 0.001, num_epochs: int = 3, batch_size: int = 64,
                      shuffle: bool = True, verbose: bool = False) -> None:
        """
        trains the Neural Net on the given Dataset
        :param dataset: input Data
        :param lr: learning rate of the Neural Network training process
        :param num_epochs: number of epochs
        :param batch_size: Batch Size
        :param shuffle: determines whether the content of the batches will be shuffled after the first epoch
        :param verbose: determines, whether the accuracy of the network will be printed after every epoch or only at the
        end of the training process
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_loader, test_loader = self.get_data_loaders(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)

                #forward
                scores = self(data)
                loss = criterion(scores, targets.long())

                #backward
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
            if verbose:
                print("testing training data")
                self.check_accuracy(loader=train_loader)
                print("testing test data")
                self.check_accuracy(loader=test_loader)
        if not verbose:
            print("testing training data")
            self.check_accuracy(loader=train_loader)
            print("testing test data")
            self.check_accuracy(loader=test_loader)

    def predict(self, dataset: dc.Data, batch_size: int = 64) -> List:
        """
        uses the trained Net to make predictions on a given Dataset
        :param dataset: input Data
        :param batch_size: batch_size to feed the network with
        :return: List of the predictions
        """
        #dont shuffle, so the predicions match up in their order with the order of the input data
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        result = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)

                scores = self(x)
                _, predictions = scores.max(1)
                for tens in predictions:
                    if isinstance(dataset, dc.SoccerDataSet):
                        result.append(dataset.class_names[tens.item()])
                    else:
                        result.append(tens.item())
        return result

    def visualize_predictions(self, dataset: dc.Data, pred_col_name: str) -> None:
        """
        visualizes predictions for the model on a given dataset. Predictions will be saved in the pred_col_name.
        Pictures will be saved in the folder of the dataset, under pics/Classifier
        :param dataset: dataset to do the visualization on
        :param pred_col_name: name of the generated column with the predictions
        """
        dataset.data[pred_col_name] = self.predict(dataset=dataset)
        dataset.extend_notes_by_one_line(f"Predicted classes using NN in column \"{pred_col_name}\".")
        vs.visualize_model_predictions(dataset=dataset, pred_col_name=pred_col_name)


def test_data_generation():
    data = dc.SoccerDataSet(save=False)
    """training_loader, test_loader = get_data_loaders(data)
    for idx, (data, target) in enumerate(training_loader):
        print(idx)
        print(data.shape)
        print(target.shape)
        print(data[1, :])
        break"""


def test_nn():
    #dataset = dc.SoccerDataSet(save=True)
    #dataset = dc.MaybeActualDataSet([20 for _ in range(6)])
    dataset = dc.Data.load(r"C:\Users\gerno\Programmieren\Bachelor\Data\220523_124623_MaybeActualDataSet")
    model = NNClassifier(dataset=dataset, num_epochs=500)
    #model_path = os.path.join(dataset.path, "model.pkl")
    #torch.save(model, model_path)
    #model = Classifier.load_classifier(dataset)
    """model.visualize_predictions(dataset, "predicted_classes")
    matrix = vs.get_change_matrix(dataset.data, ("classes", "predicted_classes"))
    matrix.to_csv("test.csv")"""
    dataset.data["NN_pred"] = model.predict(dataset=dataset)
    num_correct = len(dataset.data.loc[dataset.data["classes"] == dataset.data["NN_pred"]])
    total_len = len(dataset.data)
    print(f"{num_correct} / {total_len} predicted correctly ({(num_correct / total_len) * 100} %)")
    #model = Network(784, 10)
    #test_ = torch.randn(64, 784)
    #print(model(test_).shape)


def test_tree() -> None:
    """
    runs the training process and visualizes results
    """
    #members = [100 for _ in range(6)]
    #dataset = dc.MaybeActualDataSet(members)
    dataset = dc.MaybeActualDataSet.load(r"C:\Users\gerno\Programmieren\Bachelor\Data\220523_124623_MaybeActualDataSet")
    #dataset = dc.Data.load(r"D:\Gernot\Programmieren\Bachelor\Data\220428_124321_IrisDataSet")
    #dataset = dc.SoccerDataSet()
    model = TreeClassifier(dataset)
    """path = classifier.visualize_predictions(dataset=dataset, pred_col_name="test")
    classifier.visualize_tree(dataset=dataset, tree_pics_path=path)"""
    input_data = dataset.data[dataset.data_columns]
    input_dataset = dataset.clone_meta_data()
    input_dataset.take_new_data(input_data)
    dataset.data["tree_pred"] = model.predict(dataset=input_dataset)
    num_correct = len(dataset.data.loc[dataset.data["classes"] == dataset.data["tree_pred"]])
    total_len = len(dataset.data)
    print(f"{num_correct} / {total_len} predicted correctly ({(num_correct / total_len) * 100} %)")
    #print(tree_)
    #dataset.run_hics()
    """dataset = dc.MaybeActualDataSet.load("D:\\Gernot\\Programmieren\\Bachelor\\Python\\Experiments\\Data\\220226_135403_MaybeActualDataSet")
    trained_tree = dataset.load_tree()"""
    #matrix = vs.get_change_matrix(data, ("classes", "predicted_classes"))
    #print(matrix)


if __name__ == "__main__":
    test_nn()
    test_tree()
    #test_data_generation()
