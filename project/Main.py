from builtins import print
from project import Iris as iris
from project import Glass as glass
from project import Soybean as soybean
from project import BreastCancer as bc
from project import HouseVotes as votes
from project import FiveFold as ff
from project import NeuralNetwork as nn
import numpy as np
import sys

"""
This is the main driver class for Assignment#5.  @author: Tyler Sefcik
"""


def run_feedforward_backpropagation(data, hidden_layers, learning_rate, epoch):
    # Split the data set into 2/3 and 1/3
    data_train = data.sample(frac=.667)
    data_test = data.drop(data_train.index)

    neural_network = nn.NeuralNetwork(train_data=data_test, test_data=data_train)

    network = neural_network.new_neural_network(inputs=data_test.shape[1], hidden_layers=hidden_layers, possible_outputs=2)

    neural_network.train(network=network, training_data=data_train, learning_rate=learning_rate, epoch=epoch)

    prediction_list = []
    for row in data_test.values:
        prediction = neural_network.test(network=network, input_row=row)
        prediction_list.append(prediction)

    actual_class = []
    for row in data_test.values:
        actual = row[-1]
        actual_class.append(actual)

    correct = 0
    for index in range(len(actual_class)):
        if actual_class[index] == prediction_list[index]:
            correct += 1

    accuracy = (correct / len(actual_class)) * 100

    print("Accuracy: " + str(accuracy) + "%")
    return accuracy, prediction_list


# Iris
def run_iris(filename, target_class, class_wanted, iris_names, learning_rate, epoch):
    # Setup data
    iris_obj = iris.Iris()
    iris_data = iris_obj.setup_data_iris(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                         iris_names=iris_names)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    iris1, iris2, iris3, iris4, iris5 = five_fold.five_fold_sort_class(data=iris_data, sortby=target_class)

    print("Iris 0 layers")
    # Run 0 hidden layers
    a01, list01 = run_feedforward_backpropagation(iris1, 0, learning_rate, epoch)
    print("Classification on Iris 0 hidden layers fold 1:")
    print(list01)
    a02, list02 = run_feedforward_backpropagation(iris2, 0, learning_rate, epoch)
    a03, list03 = run_feedforward_backpropagation(iris3, 0, learning_rate, epoch)
    a04, list04 = run_feedforward_backpropagation(iris4, 0, learning_rate, epoch)
    a05, list05 = run_feedforward_backpropagation(iris5, 0, learning_rate, epoch)

    mean0 = np.average([a01, a02, a03, a04, a05])
    print("Mean Accuracy of Iris 0 hidden layers: " + str(mean0) + "%")
    print()

    print("Iris 1 hidden layer")
    # Run 1 hidden layers
    a11, list11 = run_feedforward_backpropagation(iris1, 1, learning_rate, epoch)
    print("Classification on Iris 1 hidden layers fold 1:")
    print(list11)
    a12, list12 = run_feedforward_backpropagation(iris2, 1, learning_rate, epoch)
    a13, list13 = run_feedforward_backpropagation(iris3, 1, learning_rate, epoch)
    a14, list14 = run_feedforward_backpropagation(iris4, 1, learning_rate, epoch)
    a15, list15 = run_feedforward_backpropagation(iris5, 1, learning_rate, epoch)

    mean1 = np.average([a11, a12, a13, a14, a15])
    print("Mean Accuracy of Iris 1 hidden layers: " + str(mean1) + "%")
    print()

    print("Iris 2 hidden layers")
    # Run 2 hidden layers
    a21, list21 = run_feedforward_backpropagation(iris1, 2, learning_rate, epoch)
    print("Classification on Iris 2 hidden layers fold 1:")
    print(list21)
    a22, list22 = run_feedforward_backpropagation(iris2, 2, learning_rate, epoch)
    a23, list23 = run_feedforward_backpropagation(iris3, 2, learning_rate, epoch)
    a24, list24 = run_feedforward_backpropagation(iris4, 2, learning_rate, epoch)
    a25, list25 = run_feedforward_backpropagation(iris5, 2, learning_rate, epoch)

    mean2 = np.average([a21, a22, a23, a24, a25])
    print("Mean Accuracy of Iris 2 hidden layers: " + str(mean2) + "%")
    print()


# Glass
def run_glass(filename, target_class, class_wanted, glass_names, learning_rate, epoch):
    # Setup data
    glass_obj = glass.Glass()
    glass_data = glass_obj.setup_data_glass(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                            glass_names=glass_names)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    glass1, glass2, glass3, glass4, glass5 = five_fold.five_fold_sort_class(data=glass_data, sortby=target_class)

    print("Glass 0 layers")
    # Run 0 hidden layers
    a01, list01 = run_feedforward_backpropagation(glass1, 0, learning_rate, epoch)
    print("Classification on Glass 0 hidden layers fold 1:")
    print(list01)
    a02, list02 = run_feedforward_backpropagation(glass2, 0, learning_rate, epoch)
    a03, list03 = run_feedforward_backpropagation(glass3, 0, learning_rate, epoch)
    a04, list04 = run_feedforward_backpropagation(glass4, 0, learning_rate, epoch)
    a05, list05 = run_feedforward_backpropagation(glass5, 0, learning_rate, epoch)

    mean0 = np.average([a01, a02, a03, a04, a05])
    print("Mean Accuracy of Glass 0 hidden layers: " + str(mean0) + "%")
    print()

    print("Glass 1 hidden layer")
    # Run 1 hidden layers
    a11, list11 = run_feedforward_backpropagation(glass1, 1, learning_rate, epoch)
    print("Classification on Glass 1 hidden layers fold 1:")
    print(list11)
    a12, list12 = run_feedforward_backpropagation(glass2, 1, learning_rate, epoch)
    a13, list13 = run_feedforward_backpropagation(glass3, 1, learning_rate, epoch)
    a14, list14 = run_feedforward_backpropagation(glass4, 1, learning_rate, epoch)
    a15, list15 = run_feedforward_backpropagation(glass5, 1, learning_rate, epoch)

    mean1 = np.average([a11, a12, a13, a14, a15])
    print("Mean Accuracy of Glass 1 hidden layers: " + str(mean1) + "%")
    print()

    print("Glass 2 hidden layers")
    # Run 2 hidden layers
    a21, list21 = run_feedforward_backpropagation(glass1, 2, learning_rate, epoch)
    print("Classification on Glass 2 hidden layers fold 1:")
    print(list21)
    a22, list22 = run_feedforward_backpropagation(glass2, 2, learning_rate, epoch)
    a23, list23 = run_feedforward_backpropagation(glass3, 2, learning_rate, epoch)
    a24, list24 = run_feedforward_backpropagation(glass4, 2, learning_rate, epoch)
    a25, list25 = run_feedforward_backpropagation(glass5, 2, learning_rate, epoch)

    mean2 = np.average([a21, a22, a23, a24, a25])
    print("Mean Accuracy of Glass 2 hidden layers: " + str(mean2) + "%")
    print()


# Soybean
def run_soybean(filename, target_class, learning_rate, epoch):
    # Setup data
    soybean_obj = soybean.Soybean()
    soybean_data = soybean_obj.setup_data_soybean(filename=filename, target_class=target_class)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    soybean1, soybean2, soybean3, soybean4, soybean5 = five_fold.five_fold_sort_class(data=soybean_data,
                                                                                      sortby=target_class)

    print("Soybean 0 layers")
    # Run 0 hidden layers
    a01, list01 = run_feedforward_backpropagation(soybean1, 0, learning_rate, epoch)
    print("Classification on Soybean 0 hidden layers fold 1:")
    print(list01)
    a02, list02 = run_feedforward_backpropagation(soybean2, 0, learning_rate, epoch)
    a03, list03 = run_feedforward_backpropagation(soybean3, 0, learning_rate, epoch)
    a04, list04 = run_feedforward_backpropagation(soybean4, 0, learning_rate, epoch)
    a05, list05 = run_feedforward_backpropagation(soybean5, 0, learning_rate, epoch)

    mean0 = np.average([a01, a02, a03, a04, a05])
    print("Mean Accuracy of Soybean 0 hidden layers: " + str(mean0) + "%")
    print()

    print("Soybean 1 hidden layer")
    # Run 1 hidden layers
    a11, list11 = run_feedforward_backpropagation(soybean1, 1, learning_rate, epoch)
    print("Classification on Soybean 1 hidden layers fold 1:")
    print(list11)
    a12, list12 = run_feedforward_backpropagation(soybean2, 1, learning_rate, epoch)
    a13, list13 = run_feedforward_backpropagation(soybean3, 1, learning_rate, epoch)
    a14, list14 = run_feedforward_backpropagation(soybean4, 1, learning_rate, epoch)
    a15, list15 = run_feedforward_backpropagation(soybean5, 1, learning_rate, epoch)

    mean1 = np.average([a11, a12, a13, a14, a15])
    print("Mean Accuracy of Soybean 1 hidden layers: " + str(mean1) + "%")
    print()

    print("Soybean 2 hidden layers")
    # Run 2 hidden layers
    a21, list21 = run_feedforward_backpropagation(soybean1, 2, learning_rate, epoch)
    print("Classification on Soybean 2 hidden layers fold 1:")
    print(list21)
    a22, list22 = run_feedforward_backpropagation(soybean2, 2, learning_rate, epoch)
    a23, list23 = run_feedforward_backpropagation(soybean3, 2, learning_rate, epoch)
    a24, list24 = run_feedforward_backpropagation(soybean4, 2, learning_rate, epoch)
    a25, list25 = run_feedforward_backpropagation(soybean5, 2, learning_rate, epoch)

    mean2 = np.average([a21, a22, a23, a24, a25])
    print("Mean Accuracy of Soybean 2 hidden layers: " + str(mean2) + "%")
    print()


# Breast Cancer
def run_bc(filename, target_class, class_wanted, bc_names, learning_rate, epoch):
    # Setup data
    bc_obj = bc.BreastCancer()
    bc_data = bc_obj.setup_data_bc(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                   bc_names=bc_names)
    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    bc1, bc2, bc3, bc4, bc5 = five_fold.five_fold_sort_class(data=bc_data, sortby=target_class)

    print("Breast Cancer 0 layers")
    # Run 0 hidden layers
    a01, list01 = run_feedforward_backpropagation(bc1, 0, learning_rate, epoch)
    print("Classification on Breast Cancer 0 hidden layers fold 1:")
    print(list01)
    a02, list02 = run_feedforward_backpropagation(bc2, 0, learning_rate, epoch)
    a03, list03 = run_feedforward_backpropagation(bc3, 0, learning_rate, epoch)
    a04, list04 = run_feedforward_backpropagation(bc4, 0, learning_rate, epoch)
    a05, list05 = run_feedforward_backpropagation(bc5, 0, learning_rate, epoch)

    mean0 = np.average([a01, a02, a03, a04, a05])
    print("Mean Accuracy of Breast Cancer 0 hidden layers: " + str(mean0) + "%")
    print()

    print("Breast Cancer 1 hidden layer")
    # Run 1 hidden layers
    a11, list11 = run_feedforward_backpropagation(bc1, 1, learning_rate, epoch)
    print("Classification on Breast Cancer 1 hidden layers fold 1:")
    print(list11)
    a12, list12 = run_feedforward_backpropagation(bc2, 1, learning_rate, epoch)
    a13, list13 = run_feedforward_backpropagation(bc3, 1, learning_rate, epoch)
    a14, list14 = run_feedforward_backpropagation(bc4, 1, learning_rate, epoch)
    a15, list15 = run_feedforward_backpropagation(bc5, 1, learning_rate, epoch)

    mean1 = np.average([a11, a12, a13, a14, a15])
    print("Mean Accuracy of Breast Cancer 1 hidden layers: " + str(mean1) + "%")
    print()

    print("Breast Cancer 2 hidden layers")
    # Run 2 hidden layers
    a21, list21 = run_feedforward_backpropagation(bc1, 2, learning_rate, epoch)
    print("Classification on Breast Cancer 2 hidden layers fold 1:")
    print(list21)
    a22, list22 = run_feedforward_backpropagation(bc2, 2, learning_rate, epoch)
    a23, list23 = run_feedforward_backpropagation(bc3, 2, learning_rate, epoch)
    a24, list24 = run_feedforward_backpropagation(bc4, 2, learning_rate, epoch)
    a25, list25 = run_feedforward_backpropagation(bc5, 2, learning_rate, epoch)

    mean2 = np.average([a21, a22, a23, a24, a25])
    print("Mean Accuracy of Breast Cancer 2 hidden layers: " + str(mean2) + "%")
    print()


# House Votes
def run_votes(filename, target_class, class_wanted, vote_names, learning_rate, epoch):
    # Setup data
    votes_obj = votes.HouseVotes()
    votes_data = votes_obj.setup_data_votes(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                            vote_names=vote_names)
    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    votes1, votes2, votes3, votes4, votes5 = five_fold.five_fold_sort_class(data=votes_data, sortby=target_class)

    print("House Votes 0 layers")
    # Run 0 hidden layers
    a01, list01 = run_feedforward_backpropagation(votes1, 0, learning_rate, epoch)
    print("Classification on House Votes 0 hidden layers fold 1:")
    print(list01)
    a02, list02 = run_feedforward_backpropagation(votes2, 0, learning_rate, epoch)
    a03, list03 = run_feedforward_backpropagation(votes3, 0, learning_rate, epoch)
    a04, list04 = run_feedforward_backpropagation(votes4, 0, learning_rate, epoch)
    a05, list05 = run_feedforward_backpropagation(votes5, 0, learning_rate, epoch)

    mean0 = np.average([a01, a02, a03, a04, a05])
    print("Mean Accuracy of House Votes 0 hidden layers: " + str(mean0) + "%")
    print()

    print("House Votes 1 hidden layer")
    # Run 1 hidden layers
    a11, list11 = run_feedforward_backpropagation(votes1, 1, learning_rate, epoch)
    print("Classification on House Votes 1 hidden layers fold 1:")
    print(list11)
    a12, list12 = run_feedforward_backpropagation(votes2, 1, learning_rate, epoch)
    a13, list13 = run_feedforward_backpropagation(votes3, 1, learning_rate, epoch)
    a14, list14 = run_feedforward_backpropagation(votes4, 1, learning_rate, epoch)
    a15, list15 = run_feedforward_backpropagation(votes5, 1, learning_rate, epoch)

    mean1 = np.average([a11, a12, a13, a14, a15])
    print("Mean Accuracy of House Votes 1 hidden layers: " + str(mean1) + "%")
    print()

    print("House Votes 2 hidden layers")
    # Run 2 hidden layers
    a21, list21 = run_feedforward_backpropagation(votes1, 2, learning_rate, epoch)
    print("Classification on House Votes 2 hidden layers fold 1:")
    print(list21)
    a22, list22 = run_feedforward_backpropagation(votes2, 2, learning_rate, epoch)
    a23, list23 = run_feedforward_backpropagation(votes3, 2, learning_rate, epoch)
    a24, list24 = run_feedforward_backpropagation(votes4, 2, learning_rate, epoch)
    a25, list25 = run_feedforward_backpropagation(votes5, 2, learning_rate, epoch)

    mean2 = np.average([a21, a22, a23, a24, a25])
    print("Mean Accuracy of House Votes 2 hidden layers: " + str(mean2) + "%")
    print()


# Main driver to run all algorithms on each dataset
def main():
    # Print all output to file
    # Comment out for printing in console
    # sys.stdout = open("./Assignment6Output.txt", "w")

    ##### Iris #####
    iris_target_class = "class"
    class_wanted = "Iris-virginica"
    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    run_iris(filename="data/iris.data", target_class=iris_target_class, class_wanted=class_wanted,
             iris_names=iris_names, learning_rate=.21, epoch=50)

    print()

    ##### Glass #####
    glass_target_class = "Type of glass"
    class_wanted = 3
    glass_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]
    run_glass(filename="data/glass.data", target_class=glass_target_class, class_wanted=class_wanted,
              glass_names=glass_names, learning_rate=.1, epoch=30)

    print()

    ##### Soybean #####
    soybean_target_class = "34"
    run_soybean(filename="data/soybean-small.data", target_class=soybean_target_class, learning_rate=.1, epoch=100)

    print()

    ##### Breast Cancer #####
    bc_target_class = "Class"
    bc_class_wanted = 4
    breast_cancer_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
                           "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                           "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    run_bc(filename="data/breast-cancer-wisconsin.data", target_class=bc_target_class,
                 class_wanted=bc_class_wanted, bc_names=breast_cancer_names, learning_rate=.5, epoch=100)

    print()

    ##### House Votes #####
    votes_target_class = "class"
    votes_class_wanted = "republican"
    votes_names = ["class", "handicapped", "water", "adoption", "physician", "el-salvador", "religious",
                   "anti", "aid", "mx", "immigration", "synfuels", "education", "superfund", "crime",
                   "duty-free", "export"]
    run_votes(filename="data/house-votes-84.data", target_class=votes_target_class,
                       class_wanted=votes_class_wanted, vote_names=votes_names, learning_rate=.1, epoch=50)
    print()


if __name__ == "__main__":
    main()
