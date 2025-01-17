import numpy as np
from scipy.stats import norm

class DataGenerator:
    def __init__(self, logMoneynessRange=None, maturityRange=None, volatilityRange=None, numberOfPoints=100):
        self.logMoneynessRange = np.linspace(logMoneynessRange[0], logMoneynessRange[1], numberOfPoints)
        self.maturityRange = np.linspace(maturityRange[0], maturityRange[1], numberOfPoints)
        self.volatilityRange = np.linspace(volatilityRange[0], volatilityRange[1], numberOfPoints)
        self.optionType = np.array([1, -1])  # Call (1) and put (-1)
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

        # Repeat optionType to match the size of the grids
        optionType = np.tile(self.optionType, logMoneyness.shape[0] // len(self.optionType))

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
                Input features: [log(F/K), T, sigma, optionType].
            y : np.ndarray
                Targets: [Delta, Gamma, Vega, Theta].
        """
        if self.data is None or self.greeks is None:
            raise ValueError("Data and Greeks have not been generated. Call generateTargetSpace() and generateInitialSpace() first.")
        
        # Features (X) and Targets (y)
        y = self.data  # Inputs: log(F/K), T, sigma, optionType
        x = self.greeks  # Targets: Delta, Gamma, Vega, Theta
        return x, y


if __name__ == '__main__':
    # Instantiate the data generator
    generator = DataGenerator(
        logMoneynessRange=[-1, 1], maturityRange=[0.1, 2], volatilityRange=[0.1, 0.5], numberOfPoints=5
    )
    
    # Generate data and Greeks
    generator.generateTargetSpace()
    generator.generateInitialSpace()
    
    # Prepare data for neural network
    X, y = generator.get_data_for_nn()
    
    print("Input Features (X):")
    print(X[:5])  # Display first 5 samples
    print("\nTarget Outputs (y):")
    print(y[:5])  # Display first 5 target rows
    print("\nShapes:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
