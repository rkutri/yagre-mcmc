# TODO: introduce an observer for chain diagnostics, which lets the chain
#       react dynamically to changes in the quality of samples. E.g.~flush
#       the adaptive covariance, or make the steps size smaller if the
#       acceptance rate is too low. If we store the empirical covariance
#       eigenvalues we may even set the regularisation parameter accordingly.
