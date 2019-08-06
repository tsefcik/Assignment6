import pandas as pd
from sklearn import preprocessing


class Soybean:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    def setup_data_soybean(self, filename, target_class):
        soybean_names = []
        # Fill soybean_names with column indexes
        for number in range(0, 36):
            soybean_names.append(str(number))

        # Read in data file and turn into data structure
        soybean = pd.read_csv(filename,
                              sep=",",
                              header=0,
                              names=soybean_names)

        # Get copy of data with columns that will be normalized
        new_soybean = soybean[soybean.columns[0:35]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        soybean_scaled_data = scaler.fit_transform(new_soybean)
        # Remove "class" column for now since that column will not be normalized
        soybean_names.remove(target_class)
        soybean_scaled_data = pd.DataFrame(soybean_scaled_data, columns=soybean_names)
        # Add "class" column back to our column list
        soybean_names.append(target_class)

        # Add "class" column into normalized data structure, then categorize it into integers
        soybean_scaled_data[target_class] = soybean[[target_class]]

        # Get mean of each column that will help determine what binary value to turn each into
        soybean_means = soybean_scaled_data.mean()

        # Make categorical column a binary for the class we want to use
        for index, row in soybean_scaled_data.iterrows():
            for column in soybean_names:
                # If the data value is greater than the mean of the column, make it a 1
                if soybean_scaled_data[column][index] > soybean_means[column]:
                    soybean_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    soybean_scaled_data.at[index, column] = 0

        # Return one hot encoded data frame
        return soybean_scaled_data
