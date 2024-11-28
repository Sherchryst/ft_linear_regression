import csv
import os

class DataManager:
    """This class is responsible for loading data from a CSV file and saving/loading the model parameters."""
    def __init__(self, data_file='data.csv', model_file='model.txt'):
        self.data_file = data_file
        self.model_file = model_file

    def load_data(self, *columns):
        """
        Loads data from a CSV file and returns a dictionary with the specified columns.

        Args:
            *columns (str): Names of the columns to load from the CSV file.

        Returns:
            dict: A dictionary where keys are column names and values are lists of data.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            KeyError: If one or more specified columns are not found in the file.
        """
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File {self.data_file} not found.")

        # Initialize a dictionary to store the data for each column
        data = {col: [] for col in columns}

        # Read the CSV file and populate the dictionary
        with open(self.data_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for col in columns:
                    if col not in row:
                        raise KeyError(f"Column '{col}' not found in the CSV file.")
                    data[col].append(float(row[col]))

        return data


    def save_model(self, theta0, theta1):
        """
        Saves the model parameters to a text file.

        Args:
            theta0 (float): The intercept parameter of the model.
            theta1 (float): The slope parameter of the model.
        """
        with open(self.model_file, 'w') as file:
            file.write(f"{theta0}\n{theta1}")

    def load_model(self):
        """
        Loads the model parameters from a text file.

        Returns:
            tuple: A tuple containing the intercept and slope parameters of the model.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Fichier {self.model_file} introuvable.")
        with open(self.model_file, 'r') as file:
            theta0 = float(file.readline().strip())
            theta1 = float(file.readline().strip())
        return theta0, theta1
