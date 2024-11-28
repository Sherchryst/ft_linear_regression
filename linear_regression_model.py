import logging
from data_manager import DataManager
import math

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LinearRegressionModel:
    def __init__(self, data_manager: DataManager, learning_rate=0.001, iterations=1000):
        """
        Initialize the Linear Regression model.

        Args:
            data_manager (DataManager): Instance of the DataManager class for managing file operations.
            learning_rate (float): The step size used in gradient descent to update parameters.
            iterations (int): Number of iterations to perform during gradient descent.

        Attributes:
            theta0 (float): The intercept of the linear regression model (initialized to 0.0).
            theta1 (float): The slope of the linear regression model (initialized to 0.0).
            logger (Logger): Logger instance for logging important events during training, prediction, etc.
        """
        self.data_manager = data_manager
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta0 = 0.0  # Initial value of the intercept
        self.theta1 = 0.0  # Initial value of the slope
        self.logger = logging.getLogger(self.__class__.__name__)  # Set up a logger for this class

    def train(self, mileage, price):
        """
        Train the model using gradient descent to minimize the error between predictions and actual prices.

        Args:
            mileage (list[float]): A list of car mileages (independent variable).
            price (list[float]): A list of car prices (dependent variable).

        Raises:
            ValueError: If no data points are provided or if `theta0` and `theta1` become invalid (NaN).

        Process:
            1. Normalize the `mileage` and `price` data to improve numerical stability.
            2. Iteratively update `theta0` and `theta1` using the gradient descent formulas:
                - tmp_theta0 = learningRate * (1/m) * sum(estimatePrice - price)
                - tmp_theta1 = learningRate * (1/m) * sum((estimatePrice - price) * mileage)
            3. Simultaneously update `theta0` and `theta1` to ensure correctness.
            4. Log intermediate results and check for divergence (e.g., `NaN` values in parameters).
            5. Unnormalize the parameters after training to return to the original scale.
        """
        m = len(mileage)  # Number of data points
        if m == 0:
            self.logger.error("Training data is empty.")
            raise ValueError("No data points available for training.")

        # Normalize the data for better stability during gradient descent
        mileage_mean = sum(mileage) / m
        mileage_std = (sum((x - mileage_mean) ** 2 for x in mileage) / m) ** 0.5
        price_mean = sum(price) / m
        price_std = (sum((y - price_mean) ** 2 for y in price) / m) ** 0.5

        normalized_mileage = [(x - mileage_mean) / mileage_std for x in mileage]
        normalized_price = [(y - price_mean) / price_std for y in price]

        self.logger.info("Starting training...")
        for iteration in range(self.iterations):
            # Compute predictions for all normalized mileage values
            predictions = [self._estimate_price(normalized_mileage[i]) for i in range(m)]

            # Calculate gradients for `theta0` and `theta1` using the provided formulas
            tmp_theta0 = self.learning_rate * sum(
                (predictions[i] - normalized_price[i]) for i in range(m)
            ) / m
            tmp_theta1 = self.learning_rate * sum(
                (predictions[i] - normalized_price[i]) * normalized_mileage[i] for i in range(m)
            ) / m

            # Log gradients for debugging purposes
            if iteration == 0 or iteration % (self.iterations // 10) == 0:
                self.logger.debug(f"Gradients - theta0: {tmp_theta0:.4f}, theta1: {tmp_theta1:.4f}")

            # Simultaneously update `theta0` and `theta1`
            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1

            # Log intermediate progress at every 10% of iterations
            if iteration % (self.iterations // 10) == 0:
                self.logger.info(
                    f"Iteration {iteration}: theta0={self.theta0:.4f}, theta1={self.theta1:.4f}"
                )

            # Check for `NaN` in parameters and terminate if detected
            if any(map(math.isnan, [self.theta0, self.theta1])):
                self.logger.error("Training failed: theta values became NaN. Check learning rate or data.")
                raise ValueError("NaN detected in model parameters during training.")

        # Unnormalize parameters to match the original scale of the data
        self.theta1 = self.theta1 * price_std / mileage_std
        self.theta0 = price_mean - self.theta1 * mileage_mean

        self.logger.info(f"Training completed: theta0={self.theta0:.4f}, theta1={self.theta1:.4f}")

    def predict(self, mileage):
        """
        Predict the price of a car based on its mileage using the trained model.

        Args:
            mileage (float): The mileage of the car.

        Returns:
            float: The predicted price of the car.

        Raises:
            ValueError: If the model parameters are invalid (e.g., NaN or not trained).
        """
        if not self._is_model_trained():
            self.logger.error("Model parameters are not valid. Load or train the model first.")
            raise ValueError("Model parameters are invalid. Cannot predict.")

        # Compute the predicted price using the linear regression formula
        price = self.theta0 + self.theta1 * mileage
        self.logger.info(f"Predicted price for mileage {mileage} km: {price:.2f}")
        return price

    def save_model(self):
        """
        Save the trained model parameters (`theta0` and `theta1`) to a file using DataManager.

        Raises:
            ValueError: If the model parameters are invalid (e.g., NaN).
        """
        if not self._is_model_trained():
            self.logger.error("Model parameters are invalid. Cannot save the model.")
            raise ValueError("Invalid model parameters. Cannot save.")

        # Delegate saving to DataManager
        self.data_manager.save_model(self.theta0, self.theta1)
        self.logger.info(f"Model saved successfully: theta0={self.theta0:.4f}, theta1={self.theta1:.4f}")

    def load_model(self):
        """
        Load the model parameters (`theta0` and `theta1`) from a file using DataManager.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        try:
            self.theta0, self.theta1 = self.data_manager.load_model()
            self.logger.info(f"Model loaded successfully: theta0={self.theta0:.4f}, theta1={self.theta1:.4f}")
        except FileNotFoundError as e:
            self.logger.error(str(e))
            raise e

    def _estimate_price(self, mileage):
        """
        Internal method to calculate the estimated price based on current parameters.

        Args:
            mileage (float): A normalized mileage value.

        Returns:
            float: The estimated price.
        """
        return self.theta0 + self.theta1 * mileage

    def _is_model_trained(self):
        """
        Check if the model parameters (`theta0` and `theta1`) are valid (i.e., not NaN or None).

        Returns:
            bool: True if the model is trained and parameters are valid, False otherwise.
        """
        return self.theta0 is not None and self.theta1 is not None
