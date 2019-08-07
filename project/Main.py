from builtins import print
from project import Iris as iris
from project import Glass as glass
from project import Soybean as soybean
from project import BreastCancer as bc
from project import HouseVotes as votes
from project import FiveFold as ff
from project import NeuralNetwork as nn
from project import ForwardPropagate as fp
import sys

"""
This is the main driver class for Assignment#5.  @author: Tyler Sefcik
"""


# Iris
def run_iris(filename, target_class, class_wanted, iris_names):
    # Setup data
    iris_obj = iris.Iris()
    iris_data = iris_obj.setup_data_iris(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                         iris_names=iris_names)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    iris1, iris2, iris3, iris4, iris5 = five_fold.five_fold_sort_class(data=iris_data, sortby=target_class)
    print(iris1)
    iris_network = nn.NeuralNetwork(train_data=iris1, test_data=iris2)
    iris1_network = iris_network.new_neural_network(inputs=iris1.shape[1], hidden_layers=1, possible_outputs=2)
    iris_forward = fp.ForwardPropagate()

    for row in iris1.values:
        iris_forward.forward_propagate_network(input_row=row, network=iris1_network)

    return iris1


# Glass
def run_glass(filename, target_class, class_wanted, glass_names):
    # Setup data
    glass_obj = glass.Glass()
    glass_data = glass_obj.setup_data_glass(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                            glass_names=glass_names)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    glass1, glass2, glass3, glass4, glass5 = five_fold.five_fold_sort_class(data=glass_data, sortby=target_class)

    return glass1


# Soybean
def run_soybean(filename, target_class):
    # Setup data
    soybean_obj = soybean.Soybean()
    soybean_data = soybean_obj.setup_data_soybean(filename=filename, target_class=target_class)

    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    soybean1, soybean2, soybean3, soybean4, soybean5 = five_fold.five_fold_sort_class(data=soybean_data,
                                                                                      sortby=target_class)

    return soybean1


# Breast Cancer
def run_bc(filename, target_class, class_wanted, bc_names):
    # Setup data
    bc_obj = bc.BreastCancer()
    bc_data = bc_obj.setup_data_bc(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                   bc_names=bc_names)
    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    bc1, bc2, bc3, bc4, bc5 = five_fold.five_fold_sort_class(data=bc_data, sortby=target_class)

    return bc1


# House Votes
def run_votes(filename, target_class, class_wanted, vote_names):
    # Setup data
    votes_obj = votes.HouseVotes()
    votes_data = votes_obj.setup_data_votes(filename=filename, target_class=target_class, class_wanted=class_wanted,
                                            vote_names=vote_names)
    # Setup five fold cross validation
    five_fold = ff.FiveFold()
    votes1, votes2, votes3, votes4, votes5 = five_fold.five_fold_sort_class(data=votes_data, sortby=target_class)

    return votes1


# Main driver to run all algorithms on each dataset
def main():
    # Print all output to file
    # Comment out for printing in console
    # sys.stdout = open("./Assignment6Output.txt", "w")

    ##### Iris #####
    iris_target_class = "class"
    class_wanted = "Iris-virginica"
    iris_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    iris1 = run_iris(filename="data/iris.data", target_class=iris_target_class, class_wanted=class_wanted,
                     iris_names=iris_names)

    print("Iris")
    print()

    ##### Glass #####
    glass_target_class = "Type of glass"
    class_wanted = 3
    glass_names = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]
    glass1 = run_glass(filename="data/glass.data", target_class=glass_target_class,
                       class_wanted=class_wanted, glass_names=glass_names)

    print("Glass")
    print()

    ##### Soybean #####
    soybean_target_class = "34"
    soybean1 = run_soybean(filename="data/soybean-small.data", target_class=soybean_target_class)

    print("Soybean")
    print()

    ##### Breast Cancer #####
    bc_target_class = "Class"
    bc_class_wanted = 4
    breast_cancer_names = ["Sample code number", "Clump Thickness", "Uniformity of Cell Size",
                           "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
                           "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    bc1 = run_bc(filename="data/breast-cancer-wisconsin.data", target_class=bc_target_class,
                 class_wanted=bc_class_wanted, bc_names=breast_cancer_names)

    print("Breast Cancer")
    print()

    ##### House Votes #####
    votes_target_class = "class"
    votes_class_wanted = "republican"
    votes_names = ["class", "handicapped", "water", "adoption", "physician", "el-salvador", "religious",
                   "anti", "aid", "mx", "immigration", "synfuels", "education", "superfund", "crime",
                   "duty-free", "export"]
    votes1 = run_votes(filename="data/house-votes-84.data", target_class=votes_target_class,
                       class_wanted=votes_class_wanted, vote_names=votes_names)
    print("House Votes")
    print()


if __name__ == "__main__":
    main()
