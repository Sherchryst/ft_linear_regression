# **README: Linear Regression Model**

## **Project Overview**

This project implements a simple linear regression algorithm to predict the price of a car based on its mileage. It includes functionality for training the model, estimating prices, and bonus features such as plotting the regression line and calculating model precision.

The project consists of four main scripts:

1. **`main_train.py`**: For training the linear regression model.
2. **`main_estimate.py`**: For estimating the price of a car using a trained model.
3. **`main_bonus.py`**: For bonus features such as plotting and precision calculation.
4. **`linear_regression_model.py`**: Implements the core logic for linear regression.
5. **`data_manager.py`**: A utility class for managing data and saving/loading model parameters.

---

## **Files Overview**

### 1. **`main_train.py`**
   - **Purpose**: Train the linear regression model using provided data and save the trained parameters.
   - **Usage**:
     - Accepts a CSV file containing mileage (`km`) and price (`price`) columns.
     - Trains the model using gradient descent for a specified number of iterations.
     - Saves the trained parameters (`theta0` and `theta1`) to a file.

   **Functions**:
   - `train_model(data_file, model_file, learning_rate, iterations)`:
     - Trains the model and saves the parameters.
   - Command-line interface allows running the training process with customizable parameters.

   **Example Command**:
   ```bash
   python main_train.py train --data-file data.csv --model-file model.txt --learning-rate 0.01 --iterations 1000
   ```

---

### 2. **`main_estimate.py`**
   - **Purpose**: Use a trained model to estimate the price of a car based on its mileage.
   - **Usage**:
     - Loads trained parameters from a model file.
     - Accepts a mileage value and outputs the estimated price.

   **Functions**:
   - `estimate_price(model_file, mileage)`:
     - Loads the model and estimates the price for a given mileage.
   - Command-line interface allows easy interaction for price estimation.

   **Example Command**:
   ```bash
   python main_estimate.py estimate --model-file model.txt --mileage 15000
   ```

---

### 3. **`main_bonus.py`**
   - **Purpose**: Provides bonus features such as plotting the data and regression line, and calculating model precision.
   - **Bonus Features**:
     - **Plotting**: Visualize data points and the regression line to evaluate the model.
     - **Precision Calculation**: Calculate the model's accuracy using Mean Squared Error (MSE).

   **Functions**:
   - `plot_data_and_regression_line(data_file, model_file)`:
     - Plots mileage vs price as a scatter plot, with the regression line overlayed.
   - `calculate_precision(data_file, model_file)`:
     - Calculates the Mean Squared Error (MSE) to evaluate the model's prediction accuracy.

   **Commands**:
   - **Plot the Regression Line**:
     ```bash
     python main_bonus.py plot --data-file data.csv --model-file model.txt
     ```
   - **Calculate Precision**:
     ```bash
     python main_bonus.py precision --data-file data.csv --model-file model.txt
     ```

---

### 4. **`linear_regression_model.py`**
   - **Purpose**: Implements the core logic for the linear regression algorithm.
   - **Key Features**:
     - **Gradient Descent**: Optimizes the model parameters (`theta0` and `theta1`) using the training data.
     - **Normalization**: Normalizes data to improve numerical stability during training.
     - **Unnormalization**: Converts normalized parameters back to the original scale after training.
     - **Prediction**: Predicts prices for a given mileage using trained parameters.
   - **Functions**:
     - `train(mileage, price)`: Trains the model using mileage and price data.
     - `predict(mileage)`: Predicts the price of a car for a given mileage.
     - `save_model()`: Saves the trained parameters to a file.
     - `load_model()`: Loads previously saved parameters from a file.

---

### 5. **`data_manager.py`**
   - **Purpose**: A utility class for managing data and saving/loading model parameters.
   - **Key Features**:
     - Handles reading data from CSV files.
     - Saves and loads model parameters (`theta0` and `theta1`).
   - **Functions**:
     - `load_data(columns: list)`: Reads specified columns from the CSV file.
     - `save_model(theta0, theta1)`: Saves trained model parameters to a file.
     - `load_model()`: Loads previously saved model parameters.

---

## **How to Use**

### **1. Train the Model**
   To train the linear regression model:
   ```bash
   python main_train.py train --data-file data.csv --model-file model.txt --learning-rate 0.01 --iterations 1000
   ```

   **Output**:
   - The trained model parameters (`theta0` and `theta1`) are saved in `model.txt`.

---

### **2. Estimate the Price**
   To estimate the price of a car for a given mileage:
   ```bash
   python main_estimate.py estimate --model-file model.txt --mileage 15000
   ```

   **Output**:
   - Displays the estimated price based on the given mileage.

---

### **3. Bonus Features**
   #### Plot the Data and Regression Line
   ```bash
   python main_bonus.py plot --data-file data.csv --model-file model.txt
   ```
   - Displays a graph with:
     - Blue scatter points for the data.
     - Red regression line for the model.

   #### Calculate Model Precision
   ```bash
   python main_bonus.py precision --data-file data.csv --model-file model.txt
   ```
   - Calculates and displays the Mean Squared Error (MSE) of the model.

---

## **Data Format**

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

## **Project Structure**

```
├── main_train.py            # Script for training the model
├── main_estimate.py         # Script for estimating prices
├── main_bonus.py            # Script for bonus features (plotting and precision)
├── data_manager.py          # Utility class for managing data
├── linear_regression_model.py # Core logic for linear regression
├── data.csv                 # Example dataset (mileage and price)
├── model.txt                # Saved model parameters after training
```

---

## **Dependencies**

- Python 3.x
- Required packages:
  - `matplotlib` (for plotting in `main_bonus.py`).

Install dependencies using:
```bash
pip install matplotlib
```

---

## **Example Workflow**

### **Step 1: Train the Model**
```bash
python main_train.py train --data-file data.csv --model-file model.txt --learning-rate 0.01 --iterations 1000
```

### **Step 2: Estimate the Price**
```bash
python main_estimate.py estimate --model-file model.txt --mileage 15000
```

### **Step 3: Bonus Features**
- Plot the Data and Regression Line:
  ```bash
  python main_bonus.py plot --data-file data.csv --model-file model.txt
  ```
- Calculate Model Precision:
  ```bash
  python main_bonus.py precision --data-file data.csv --model-file model.txt
  ```
