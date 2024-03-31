"""
Class to create a graph for the csv file that register the Average Loss
for each model given the iterations/epoch's
"""
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.interpolate import make_interp_spline


def plotLineGraph(source):
    df = pd.read_csv(source, sep=',')
    # plt.plot(df['Iteration'], df['Average Loss'], label='Average Loss')

    X_Y_Spline = make_interp_spline(df['Iteration'], df['Average Loss'])
    X_ = np.linspace(df['Iteration'].min(), df['Iteration'].max(), 500)
    Y_ = X_Y_Spline(X_)

    # Plotting the Graph
    plt.plot(X_, Y_, color="violet")
    # without Smoothing
    # plt.plot(df['Iteration'],df['Average Loss'], color="violet")
    plt.title("Average Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Average Loss")
    plt.show()


if __name__ == "__main__":
    plotLineGraph(sys.argv[1])
