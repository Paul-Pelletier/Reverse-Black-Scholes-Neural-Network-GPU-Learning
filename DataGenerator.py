import numpy as np


class DataGenerator:
    def __init__(self, logMoneynessRange=None, maturityRange=None, volatilityRange=None, numberOfPoints=100):
        self.logMoneynessRange = np.linspace(logMoneynessRange[0], logMoneynessRange[1], numberOfPoints)
        self.maturityRange = np.linspace(maturityRange[0], maturityRange[1], numberOfPoints)
        self.volatilityRange = np.linspace(volatilityRange[0], volatilityRange[1], numberOfPoints)

    def generate(self):
        #The 3 arrays are generated in the __init__ method but I want a matrix that combines as a grid the 3 arrays
        #I will use the meshgrid function from numpy
        logMoneyness, maturity, volatility = np.meshgrid(self.logMoneynessRange, self.maturityRange, self.volatilityRange)
        data = np.c_[logMoneyness.ravel(), maturity.ravel(), volatility.ravel()]
        return data

if __name__ == '__main__':
    generator = DataGenerator(logMoneynessRange=[-1, 1], maturityRange=[0.1, 2], volatilityRange=[0.1, 0.5], numberOfPoints=3)
    data = generator.generate()
    print(data)
    print(data.shape)