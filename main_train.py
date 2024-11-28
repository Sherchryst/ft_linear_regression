import logging
from data_manager import DataManager
from linear_regression_model import LinearRegressionModel

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def train_model(data_file, model_file, learning_rate=0.1, iterations=1000):
    """
    Train the linear regression model using data from a specified file and save the trained parameters.

    Args:
        data_file (str): Path to the CSV fil containing mileage and price data.
        model_file (str): Path to save the trained model parameters.
        learning_rate (float): The step size for gradient descent.
        iterations (int): Number of iterations for gradient descent.

    Returns:
        None
    """
    # Initialize DataManager and load data
    data_manager = DataManager(data_file=data_file, model_file=model_file)
    data = data_manager.load_data('km', 'price')

    mileage = data['km']
    price = data['price']

    # Initialize and train the model
    model = LinearRegressionModel(data_manager, learning_rate=learning_rate, iterations=iterations)
    model.train(mileage, price)
    model.save_model()

    logging.info("Model training completed and saved successfully.")

if __name__ == "__main__":
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Linear Regression Model: Train")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands: train")

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train the linear regression model")
    train_parser.add_argument("--data-file", required=True, help="Path to the CSV file containing mileage and price data")
    train_parser.add_argument("--model-file", required=True, help="Path to save the trained model parameters")
    train_parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for gradient descent (default: 0.001)")
    train_parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for gradient descent (default: 1000)")

    args = parser.parse_args()

    # Handle subcommands
    if args.command == "train":
        train_model(
            data_file=args.data_file,
            model_file=args.model_file,
            learning_rate=args.learning_rate,
            iterations=args.iterations
        )
    else:
        parser.print_help()
