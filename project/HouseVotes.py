import pandas as pd
from sklearn import preprocessing


class HouseVotes:
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 3000)

    def setup_data_votes(self, filename, target_class, class_wanted, vote_names):
        # Read in data file and turn into data structure
        votes = pd.read_csv(filename,
                            sep=",",
                            header=0,
                            names=vote_names)

        # Keep track of rows to drop with missing data
        rows_to_drop = []

        # Find rows with values not filled in
        for column in vote_names:
            for index, row in votes.iterrows():
                if row[column] == "?":
                    rows_to_drop.append(index)

        # Drop rows without full data
        votes = votes.drop(rows_to_drop, axis=0)

        # Make categorical column a binary for the class we want to use
        for index, row in votes.iterrows():
            if votes[target_class][index] == class_wanted:
                votes.at[index, target_class] = 1
            else:
                votes.at[index, target_class] = 0

        # Make categorical column a binary for the class we want to use
        for column in vote_names:
            for index, row in votes.iterrows():
                if votes[column][index] == "y":
                    votes.at[index, column] = 1
                else:
                    votes.at[index, column] = 0

        # Get copy of data with columns that will be normalized
        new_votes = votes[votes.columns[1:17]]
        # Normalize data with sklearn MinMaxScaler
        scaler = preprocessing.MinMaxScaler()
        votes_scaled_data = scaler.fit_transform(new_votes)
        # Remove "class" column for now since that column will not be normalized
        vote_names.remove(target_class)
        votes_scaled_data = pd.DataFrame(votes_scaled_data, columns=vote_names)
        # Add "class" column back to our column list
        vote_names.insert(0, target_class)

        # Add "class" column into normalized data structure, then categorize it into integers
        votes_scaled_data[target_class] = votes[[target_class]]

        # Get mean of each column that will help determine what binary value to turn each into
        votes_means = votes_scaled_data.mean()

        # Make categorical column a binary for the class we want to use
        for index, row in votes_scaled_data.iterrows():
            for column in vote_names:
                # If the data value is greater than the mean of the column, make it a 1
                if votes_scaled_data[column][index] > votes_means[column]:
                    votes_scaled_data.at[index, column] = 1
                # Otherwise make it a 0 since it is less than the mean
                else:
                    votes_scaled_data.at[index, column] = 0

        # Return one hot encoded data frame
        return votes_scaled_data
