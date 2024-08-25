from typing import Any
import pandas as pd
import numpy as np

from constants import PSI_BUCKET


class PSI:
    '''
    Calculate the PSI (population stability index) across all variables
    '''

    @staticmethod
    def psi(expected_fractions, actual_fractions, buckets=PSI_BUCKET) -> float:
        '''Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: the values avoids the divergence

        Returns:
           psi_value: calculated PSI value
        '''

        def sub_psi(e_perc, a_perc):
            '''
            Calculate the actual PSI value from comparing the values.
            Update the actual value to a very small number if equal to zero
            '''

            if a_perc == 0:
                a_perc = buckets
            if e_perc == 0:
                e_perc = buckets

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value
            
        psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i]) for i in range(0, len(expected_fractions)))
        
        return psi_value
    
    def __repr__(self) -> str:
        return "PSI"
    
    def run(
        self, 
        reference: pd.Series,     
        production: pd.Series, 
        buckets=PSI_BUCKET, 
        axis=0
    ) -> float:
        '''
        Args:
            - reference <pd.Series>: get probability in reference to compute
            - production <pd.Series>: get probability in production to compute
            - buckettype <str>: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
            - buckets <int>: the values avoids the divergence
            - axis <int>: axis by which variables are defined, 0 for vertical, 1 for horizontal
        
        Return:
            - psi_values <float>: return the computed PSI value
        '''

        if len(reference.shape) == 1:
            psi_values = np.empty(len(reference.shape))
        else:
            psi_values = np.empty(reference.shape[1 - axis])
        
        for i in range(0, len(psi_values)):
            if len(psi_values) == 1:
                psi_values = self.psi(reference, production, buckets)
            elif axis == 0:
                psi_values[i] = self.psi(reference[:,i], production[:,i], buckets)
            elif axis == 1:
                psi_values[i] = self.psi(reference[i,:], production[i,:], buckets)

        return psi_values