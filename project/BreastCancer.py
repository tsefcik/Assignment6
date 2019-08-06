import pandas as pd
from sklearn import preprocessing


class BreastCancer:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    def setup_data_bc(self, filename, target_class, class_wanted, bc_names):
        # Read in data file and turn into data structure
        breast_cancer = pd.read_csv(filename,
                                    sep=",",
                                    header=0,
                                    names=bc_names)

        # Keep track of rows to drop with missing data
        rows_to_drop = []

        # Find rows with values not filled in
        for column in bc_names:
            for index, row in breast_cancer.iterrows():
                if row[column] == "?":
                    rows_to_drop.append(index)

        # Drop rows without full data
        breast_cancer = breast_cancer.drop(rows_to_drop, axis=0)

        # Make categorical column a binary for the class we want to use
        for index, row in breast_cancer.iterrows():
            if breast_cancer[target_class][index] == class_wanted:
                breast_cancer.at[index, target_class] = 1
            else:
                breast_cancer.at[index, target_class] = 0

        # Get copy of data with columns that will be normalized
        new_breast_cancer = breast_cancer[breast_cancer.columns[1:10]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        breast_cancer_scaled_data = scaler.fit_transform(new_breast_cancer)
        # Remove "Class" column for now since that column will not be normalized
        bc_names.remove(target_class)
        # Remove "Sample code number" column for now since that column will not be normalized
        bc_names.remove("Sample code number")
        breast_cancer_scaled_data = pd.DataFrame(breast_cancer_scaled_data, columns=bc_names)
        # Add "class" column back to our column list
        bc_names.append(target_class)

        # Add "Class" column into normalized data structure, then categorize it into integers
        breast_cancer_scaled_data[target_class] = breast_cancer[[target_class]]

        # Get mean of each column that will help determine what binary value to turn each into
        breast_cancer_means = breast_cancer_scaled_data.mean()

        # Make categorical column a binary for the class we want to use
        for index, row in breast_cancer_scaled_data.iterrows():
            for column in bc_names:
                # If the data value is greater than the mean of the column, make it a 1
                if breast_cancer_scaled_data[column][index] > breast_cancer_means[column]:
                    breast_cancer_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    breast_cancer_scaled_data.at[index, column] = 0

        # Add column back in bc it was throwing an error on the iterations of running the winnow alg
        bc_names.insert(0, "Sample code number")

        # Return one hot encoded data frame
        return breast_cancer_scaled_data
