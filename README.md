# **README: Linear Regression Model**

## **Project Overview**

This project implements a simple linear regression algorithm to predict the price of a car based on its mileage. It consists of three main files that handle different parts of the workflow:

1. **Training the Model**: Using historical data of mileage and prices to train the model and save its parameters.
2. **Estimating Prices**: Using the trained model to predict the price of a car given its mileage.
3. **Data Management**: A utility class that handles the loading and saving of data and model parameters.

---

## **Files Overview**

### 1. **`main_train.py`**
   - **Purpose**: This script trains the linear regression model using provided data and saves the trained parameters for later use.
   - **Usage**:
     - Accepts a CSV file containing mileage (`km`) and price (`price`) columns.
     - Trains the model using gradient descent for a specified number of iterations.
     - Saves the trained parameters (`theta0` and `theta1`) to a file.

   **Functions**:
   - `train_model(data_file, model_file, learning_rate, iterations)`:
     - Trains the model and saves the parameters.
   - Command-line interface allows running the training process with customizable parameters.

---

### 2. **`main_estimate.py`**
   - **Purpose**: This script loads the pre-trained model and estimates the price of a car given its mileage.
   - **Usage**:
     - Requires the file containing the trained model parameters.
     - Accepts a mileage value as input and outputs the predicted price.

   **Functions**:
   - `estimate_price(model_file, mileage)`:
     - Loads the trained parameters and estimates the price for a given mileage.
   - Command-line interface allows estimating prices easily via terminal commands.

---

### 3. **`data_manager.py`**
   - **Purpose**: A utility class that manages loading and saving of data and model parameters.
   - **Usage**:
     - Handles reading CSV files with mileage and price data.
     - Saves and loads model parameters (`theta0` and `theta1`) for use in training and estimation.
   - **Functions**:
     - `load_data(columns: list)`: Reads specified columns from the CSV file.
     - `save_model(theta0, theta1)`: Saves trained model parameters to a file.
     - `load_model()`: Loads previously saved model parameters.

---

## **How to Use**

### **1. Training the Model**
   To train the linear regression model, use the `main_train.py` script.

   **Command**:
   ```bash
   python main_train.py train --data-file <data.csv> --model-file <model.txt> --learning-rate <learning_rate> --iterations <iterations>
   ```

   **Arguments**:
   - `--data-file`: Path to the CSV file containing mileage and price data (required).
   - `--model-file`: Path to save the trained model parameters (required).
   - `--learning-rate`: Learning rate for gradient descent (default: `0.1`).
   - `--iterations`: Number of iterations for gradient descent (default: `1000`).

   **Example**:
   ```bash
   python main_train.py train --data-file data.csv --model-file model.txt --learning-rate 0.01 --iterations 1000
   ```

   **Output**:
   - The trained model parameters (`theta0` and `theta1`) will be saved in the specified file (e.g., `model.txt`).

---

### **2. Estimating Prices**
   To estimate the price of a car given its mileage, use the `main_estimate.py` script.

   **Command**:
   ```bash
   python main_estimate.py estimate --model-file <model.txt> --mileage <mileage_value>
   ```

   **Arguments**:
   - `--model-file`: Path to the file containing trained model parameters (required).
   - `--mileage`: Mileage of the car to estimate the price for (required).

   **Example**:
   ```bash
   python main_estimate.py estimate --model-file model.txt --mileage 15000
   ```

   **Output**:
   - The estimated price for the specified mileage will be displayed.

---

### **3. Data Format**
   The data file should be a CSV file containing at least two columns:
   - **`km`**: The mileage of the car.
   - **`price`**: The price of the car.

   **Example**:
   ```
   km,price
   10000,8000
   20000,7000
   30000,6000
   ```

---

## **Project Workflow**

1. **Prepare Data**:
   - Create a CSV file with mileage and price data.

2. **Train the Model**:
   - Use `main_train.py` to train the model with the prepared data.

3. **Estimate Prices**:
   - Use `main_estimate.py` with the trained model to estimate the price of a car given its mileage.

---

## **Example Workflow**

### **Training**
```bash
python main_train.py train --data-file data.csv --model-file model.txt --learning-rate 0.01 --iterations 1000
```
- Output:
  ```
  2024-11-28 17:30:00,123 - LinearRegressionModel - INFO - Starting training...
  2024-11-28 17:31:00,456 - LinearRegressionModel - INFO - Training completed: theta0=5000.50, theta1=-20.30
  Model training completed and saved successfully.
  ```

### **Estimating Prices**
```bash
python main_estimate.py estimate --model-file model.txt --mileage 15000
```
- Output:
  ```
  2024-11-28 17:35:00,789 - LinearRegressionModel - INFO - Predicted price for mileage 15000 km: 3000.00
  Estimated price for mileage 15000 km: 3000.00
  ```

---

## **Dependencies**

- Python 3.x
- Required packages:
  - `logging`
  - `argparse`
  - Any custom utilities (`data_manager.py`, `linear_regression_model.py`).

---

## **Project Structure**

```
├── main_train.py        # Script to train the model
├── main_estimate.py     # Script to estimate prices
├── data_manager.py      # Handles loading and saving of data and model parameters
├── linear_regression_model.py # The Linear Regression implementation
├── data.csv             # Example data (mileage and price)
├── model.txt            # Saved model parameters after training
```

---

## **Future Improvements**
- Add support for more features (e.g., multiple independent variables).
- Implement error metrics like Mean Squared Error (MSE) for evaluation.
- Add unit tests for each module.

---

Let me know if you'd like any further refinements or additions!
