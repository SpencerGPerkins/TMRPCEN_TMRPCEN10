"""
@author: Spencer perkins

Analysis Script : Compute summary statistics (averages, standard deviation)
"""
import numpy as np

class Averaging():

    def overall(averages):
        """
        overall average of validation average from Training
        """
        return (np.sum(averages)/len(averages))

class STD():

    def overall_std(averages, overall_average):
        """Standard deviation of validation accuracy averages
        Params
        --------
        averages : list, column with each validation average across epochs
        overall_average : float, average over all epochs

        Returns
        --------
        Standard Deviation
        """
        N = len(averages)
        numerator = []

        for average in averages:
            x = (average-averages)**2
            numerator.append(x)
            
        return np.sqrt(np.sum(numerator)/N)
