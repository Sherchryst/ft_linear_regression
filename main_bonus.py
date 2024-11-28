import logging
from data_manager import DataManager
from linear_regression_model import LinearRegressionModel
import matplotlib.pyplot as plt

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def plot_data_and_regression_line(data_file, model_file):
    """
    Visualize the data points (mileage vs price) along with the regression line.

    Args:
        data_file (str): Path to the CSV file containing mileage and price data.
        model_file (str): Path to the file containing the trained model parameters.

    Process:
        1. Load mileage and price data from the CSV file using the DataManager.
        2. Load the trained model parameters (theta0 and theta1) from the model file.
        3. Generate a range of mileage values and compute corresponding predicted prices using the model.
        4. Plot the data points as a scatter plot.
        5. Plot the regression line using the predicted prices.
        6. Display the graph with appropriate labels, title, and legend.

    Returns:
        None. Displays a graph with data points and the regression line.
    """
    # Step 1: Load the data from the CSV file
    data_manager = DataManager(data_file=data_file, model_file=model_file)
    data = data_manager.load_data('km', 'price')

    mileage = data['km']
    price = data['price']

    # Step 2: Load the trained model
    model = LinearRegressionModel(data_manager)
    model.load_model()

    # Step 3: Generate a range of mileage values for the regression line
    min_mileage, max_mileage = min(mileage), max(mileage)
    regression_line_mileage = [
        min_mileage + i * (max_mileage - min_mileage) / 99 for i in range(100)
    ]  # Generate 100 evenly spaced values
    regression_line_price = [model.predict(m) for m in regression_line_mileage]  # Predict prices for these mileage values

    # Step 4: Plot the data points (scatter plot)
    plt.scatter(mileage, price, color='blue', label='Data points')

    # Step 5: Plot the regression line
    plt.plot(regression_line_mileage, regression_line_price, color='red', label='Regression line')

    # Step 6: Customize the graph
    plt.xlabel('Mileage (km)')  # Label for x-axis
    plt.ylabel('Price (€)')  # Label for y-axis
    plt.title('Mileage vs Price with Regression Line')
    plt.legend()
    plt.grid(True)

    # Step 7: Display the graph
    plt.show()


def calculate_precision(data_file, model_file):
    """
    Calculate the precision of the trained model using Mean Squared Error (MSE).

    Args:
        data_file (str): Path to the CSV file containing mileage and price data.
        model_file (str): Path to the file containing the trained model parameters.

    Process:
        1. Load mileage and price data from the CSV file using the DataManager.
        2. Load the trained model parameters (theta0 and theta1) from the model file.
        3. Use the model to predict the price for each mileage value in the dataset.
        4. Calculate the Mean Squared Error (MSE) as follows:
            - MSE = (1 / n) * Σ (predicted_price[i] - actual_price[i])²
              where n is the number of data points.
        5. Log the calculated MSE for transparency and debugging.

    Returns:
        float: The Mean Squared Error (MSE), a measure of the model's prediction accuracy.
    """
    # Step 1: Load the data from the CSV file
    data_manager = DataManager(data_file=data_file, model_file=model_file)
    data = data_manager.load_data('km', 'price')

    mileage = data['km']
    price = data['price']

    # Step 2: Load the trained model
    model = LinearRegressionModel(data_manager)
    model.load_model()

    # Step 3: Use the model to make predictions for the mileage values
    predictions = [model.predict(m) for m in mileage]  # List of predicted prices

    # Step 4: Calculate Mean Squared Error (MSE)
    mse = sum((predictions[i] - price[i]) ** 2 for i in range(len(price))) / len(price)

    # Step 5: Log the MSE for transparency
    logging.info(f"Model precision (MSE): {mse:.2f}")
    return mse


if __name__ == "__main__":
    import argparse

    # Set up argument parsing for the bonus features
    parser = argparse.ArgumentParser(description="Linear Regression Model: Bonus Features")
    subparsers = parser.add_subparsers(dest="command", help="Bonus commands: plot or precision")

    # Subcommand: plot
    plot_parser = subparsers.add_parser("plot", help="Plot data points and regression line")
    plot_parser.add_argument("--data-file", required=True, help="Path to the CSV file containing mileage and price data")
    plot_parser.add_argument("--model-file", required=True, help="Path to the file containing trained model parameters")

    # Subcommand: precision
    precision_parser = subparsers.add_parser("precision", help="Calculate the precision of the model")
    precision_parser.add_argument("--data-file", required=True, help="Path to the CSV file containing mileage and price data")
    precision_parser.add_argument("--model-file", required=True, help="Path to the file containing trained model parameters")

    args = parser.parse_args()

    # Handle subcommands
    if args.command == "plot":
        plot_data_and_regression_line(data_file=args.data_file, model_file=args.model_file)
    elif args.command == "precision":
        mse = calculate_precision(data_file=args.data_file, model_file=args.model_file)
        print(f"Model precision (MSE): {mse:.2f}")
    else:
        parser.print_help()
