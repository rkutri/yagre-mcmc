import numpy as np
from scipy.signal import correlate


def estimate_autocorrelation_function_1d(sequence):
    """
    Estimate the autocorrelation function of a 1D sequence.

    Parameters
    ----------
    sequence : list or np.ndarray
        The input sequence for which the autocorrelation function is estimated.

    Returns
    -------
    np.ndarray
        The smoothed autocorrelation function.
    """

    n = len(sequence)

    # center the sequence
    sequence = np.array(sequence) - np.mean(sequence)

    # compute covariances and omit 'negative lags'
    acf = correlate(sequence, sequence, mode='full', method='auto')[n - 1:]
    acf /= acf[0]

    return acf


def sokal_heuristic(iatSeq, heuristicConst):
    """
    Determine the maximum lag to consider for IAT estimation using Sokal's
    heuristic, Sokal (1989)

    Parameters
    ----------
    iatSeq : np.ndarray
        The sequence of IAT estimates.
    heuristicConst : float
        The heuristic constant used to determine the maximum lag.

    Returns
    -------
    int
        The maximum lag satisfying Sokal's heuristic, or the total sequence
        length if none satisfy it.
    """

    seqLength = len(iatSeq)

    heuristicSatisfied = np.arange(seqLength) < heuristicConst * iatSeq

    if np.any(heuristicSatisfied):
        return np.argmin(heuristicSatisfied)

    # if no lag satisfies the heuristic, return maximum lag
    return seqLength - 1


def integrated_autocorrelation_1d(seq, sokalConst=5.0):
    """
    Estimate the integrated autocorrelation time (IAT) of a sequence using
    the method outlined in Goodman & Weare (2010) and implemented as
    suggested in https://emcee.readthedocs.io/en/stable/tutorials/autocorr/.

    Parameters
    ----------
    seq : list or np.ndarray
        The one-dimensional input sequence for IAT estimation.
    sokal_const : float, optional
        The heuristic constant used to determine the maximum lag. Default is 5.0.

    Returns
    -------
    float
        The estimated integrated autocorrelation time.
    """

    seq = np.atleast_1d(seq)

    if seq.ndim != 1:
        raise ValueError("Input sequence must be one-dimensional.")

    iatSeq = 2. * np.cumsum(seq) - 1.
    maxLag = sokal_heuristic(iatSeq, sokalConst)

    return int(np.rint(iatSeq[maxLag]))


def integrated_autocorrelation_nd(seq, method='mean', sokalConst=5.):
    """
    Parameters
    ----------
    seq : list of np.ndarray
        A list of d-dimensional NumPy arrays representing the sequence
        for which the integrated autocorrelation time is to be estimated.
        Each array corresponds to a state of the sequence.

    method : str, optional
        Specifies the method for aggregating autocorrelation information
        across dimensions when d > 1. Options are:
            - 'mean': Compute the IAT for the mean across all dimensions.
            - 'max': Compute the IAT for each dimension individually and
              return the largest value.
        The 'max' option is more computationally expensive for high-dimensional
        sequences. Default is 'mean'.

    sokalConst : float, optional
        A heuristic constant used to determine the maximum lag to consider
        when estimating the IAT. Default is 5.0.

    Returns
    -------
    iat : float
        The estimated integrated autocorrelation time for the sequence.
    """

    seq = np.asarray(seq)

    if method not in ['mean', 'max']:
        raise ValueError(f"Invalid IAT - Type: {method}. Options are 'mean' "
                         "and 'max'.")

    if method == 'mean':

        meanAcf = estimate_autocorrelation_function_1d(np.mean(seq, axis=1))
        return integrated_autocorrelation_1d(meanAcf)

    elif method == 'max':

        dim = seq.shape[1]
        iatList = [integrated_autocorrelation_1d(
            estimate_autocorrelation_function_1d(seq[:, d]))
            for d in range(dim)]

        return max(iatList)
    else:
        raise RuntimeError("undefined IAT estimation method")
