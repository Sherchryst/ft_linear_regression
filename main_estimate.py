

import logging
from data_manager import DataManager
from linear_regression_model import LinearRegressionModel

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def estimate_price(model_file, mileage):
    """
    Estimate the price of a car given its mileage using a pre-trained model.

    Args:
        model_file (str): Path to the file containing trained model parameters.
        mileage (float): Mileage of the car to estimate the price for.

    Returns:
        float: The estimated price.
    """
    # Initialize DataManager and load the model
    data_manager = DataManager(model_file=model_file)
    model = LinearRegressionModel(data_manager)
    model.load_model()

    # Predict the price
    estimated_price = model.predict(mileage)
    logging.info(f"Estimated price for mileage {mileage} km: {estimated_price:.2f}")
    return estimated_price

if __name__ == "__main__":
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Linear Regression Model: Estimate")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands: estimate")

    # Subcommand: estimate
    estimate_parser = subparsers.add_parser("estimate", help="Estimate the price of a car")
    estimate_parser.add_argument("--model-file", required=True, help="Path to the file containing trained model parameters")
    estimate_parser.add_argument("--mileage", type=float, required=True, help="Mileage of the car to estimate the price for")

    args = parser.parse_args()

    # Handle subcommands
    if args.command == "estimate":
        price = estimate_price(
            model_file=args.model_file,
            mileage=args.mileage
        )
        print(f"Estimated price for mileage {args.mileage} km: {price:.2f}")
    else:
        parser.print_help()

