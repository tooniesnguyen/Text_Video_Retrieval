import pandas as pd
import numpy as np
from typing import Tuple

from constants import ZERO_PROB_SCALE, ZERO_PROB_LIMIT



def compute_probability(
    reference: pd.Series, 
    production: pd.Series
) -> Tuple[np.array, np.array]:
    '''
    Compute the probability of independent valuable X <P(X)>.

    Args:
        - reference <pd.Series>: the considered feature in reference.
        - production <pd.Series>: the considered feature in production.
    
    Return:
        - reference_percents, current_percents <np.array>: the probability of independent valuable X <P(X)> in reference and production.
    '''

    n_vals = reference.nunique()

    bins = np.histogram_bin_edges(pd.concat([reference, production], axis=0).values, bins="sturges")
    reference_percents = np.histogram(reference, bins)[0] / len(reference)
    current_percents = np.histogram(production, bins)[0] / len(production)

    np.place(
        reference_percents,
        reference_percents == 0,
        min(reference_percents[reference_percents != 0]) / ZERO_PROB_SCALE
        if min(reference_percents[reference_percents != 0]) <= ZERO_PROB_LIMIT
        else ZERO_PROB_LIMIT,
    )
    np.place(
        current_percents,
        current_percents == 0,
        min(current_percents[current_percents != 0]) / ZERO_PROB_SCALE
        if min(current_percents[current_percents != 0]) <= ZERO_PROB_LIMIT
        else ZERO_PROB_LIMIT,
    )

    return reference_percents, current_percents