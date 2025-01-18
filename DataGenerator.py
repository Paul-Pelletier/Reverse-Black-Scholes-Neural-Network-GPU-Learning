import numpy as np
from scipy.stats import norm

class DataGenerator:
    def __init__(self, logMoneynessRange=None, maturityRange=None, volatilityRange=None, numberOfPoints=100):
        self.logMoneynessRange = np.linspace(logMoneynessRange[0], logMoneynessRange[1], numberOfPoints)
        self.maturityRange = np.linspace(maturityRange[0], maturityRange[1], numberOfPoints)
        self.volatilityRange = np.linspace(volatilityRange[0], volatilityRange[1], numberOfPoints)
        self.optionType = np.array([1, -1])  # Call (1) and Put (-1)
        self.data = None
        self.greeks = None

    def generateTargetSpace(self):
        # Create a meshgrid for the first three dimensions
        logMoneyness, maturity, volatility = np.meshgrid(
            self.logMoneynessRange, self.maturityRange, self.volatilityRange, indexing="ij"
        )

        # Flatten the grids
        logMoneyness = logMoneyness.ravel()
        maturity = maturity.ravel()
        volatility = volatility.ravel()

        # Calculate the required length for optionType
        total_points = len(logMoneyness)

        # Repeat optionType to match the total number of points
        optionType = np.tile(self.optionType, total_points // len(self.optionType) + 1)[:total_points]

        # Combine all dimensions into a single matrix
        self.data = np.c_[
            logMoneyness,
            maturity,
            volatility,
            optionType
        ]
        return self.data


    @staticmethod
    def black_scholes_greeks(data):
        """
        Calculates the Greeks of an option using the Black-Scholes model.

        Parameters:
            data : np.ndarray
                Array containing [log(F/K), T, sigma, optionType] for each option.

        Returns:
            np.ndarray : Contains Delta, Gamma, Vega, and Theta as columns.
        """
        # Unpack data
        log_fk = data[:, 0]
        T = data[:, 1]
        sigma = data[:, 2]
        option_type = data[:, 3]

        # Validate inputs
        if np.any(T <= 0):
            raise ValueError("Residual maturity T must be strictly positive.")

        # Calculate d1 and d2
        d1 = (log_fk + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate Greeks
        phi_d1 = norm.pdf(d1)  # PDF
        N_d1 = norm.cdf(d1)    # CDF
        delta = np.where(option_type == 1, N_d1, N_d1 - 1)
        gamma = phi_d1 / (np.exp(log_fk) * sigma * np.sqrt(T))
        vega = np.exp(log_fk) * np.sqrt(T) * phi_d1
        theta = - (np.exp(log_fk) * phi_d1 * sigma) / (2 * np.sqrt(T))

        # Stack Greeks into a single matrix
        return np.stack([delta, gamma, vega, theta], axis=1)

    def generateInitialSpace(self):
        if self.data is None:
            raise ValueError("Target space data has not been generated yet. Call generateTargetSpace() first.")
        self.greeks = self.black_scholes_greeks(self.data)
        return self.greeks

    def get_data_for_nn(self):
        """
        Returns the dataset formatted for neural network training.

        Outputs:
            X : np.ndarray
                Input features: [Delta, Gamma, Vega, Theta, T, optionType].
            y : np.ndarray
                Targets: [log(F/K), sigma].
        """
        if self.data is None or self.greeks is None:
            raise ValueError("Data and Greeks have not been generated. Call generateTargetSpace() and generateInitialSpace() first.")

        # Features (X): [Delta, Gamma, Vega, Theta, T, optionType]
        X = np.c_[self.greeks, self.data[:, 1], self.data[:, 3]]  # Add T and optionType

        # Targets (y): [log(F/K), sigma]
        y = self.data[:, [0, 2]]
        return X, y


if __name__ == '__main__':
    # Instantiate the data generator
    generator = DataGenerator(
        logMoneynessRange=[-0.1, 0.1],
        maturityRange=[0.1, 2],
        volatilityRange=[0.1, 0.5],
        numberOfPoints=5
    )
    
    # Generate data and Greeks
    generator.generateTargetSpace()
    generator.generateInitialSpace()
    
    # Prepare data for neural network
    X, y = generator.get_data_for_nn()
    
    print("Input Features (X):")
    print(X)  # Display first 5 samples
    print("\nTarget Outputs (y):")
    print(y)  # Display first 5 target rows
    print("\nShapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
