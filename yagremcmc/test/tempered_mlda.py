
def test_tempering_update():

    class MockModel:
        def __init__(self):
            self.likelihood = MagicMock()
            self.likelihood.tempering = 1.0

        def log_likelihood(self, parameter):
            return -0.5 * parameter ** 2

        def log_prior(self, parameter):
            return -abs(parameter)

    # Create UnnormalisedPosterior instances with mock models
    nSurrogates = 3
    surrogateDensities = [UnnormalisedPosterior(
        MockModel()) for _ in range(nSurrogates)]
    finestTarget = UnnormalisedPosterior(MockModel())

    # Mock parameters for MLDAProposal
    baseProposalCov = 1.0  # Simplified for testing
    nSteps = [10, 20, 30]

    # Initialize MLDA
    mlda = MLDA(finestTarget, surrogateDensities, baseProposalCov, nSteps)

    # Define the tempering sequence
    temperingSequence1 = [0.1, 0.5, 0.9]

    # Apply the first tempering sequence
    mlda.set_tempering_sequence(temperingSequence1)

    # Verify tempering values match the first sequence
    for i, density in enumerate(surrogateDensities):
        assert density._model.likelihood.tempering == temperingSequence1[i], (
            f"Tempering for surrogate {i} did not update correctly. "
            f"Expected {temperingSequence1[i]}, got {density._model.likelihood.tempering}."
        )

    # Define a second tempering sequence
    temperingSequence2 = [0.2, 0.6, 1.0]

    # Apply the second tempering sequence
    mlda.set_tempering_sequence(temperingSequence2)

    # Verify tempering values match the second sequence
    for i, density in enumerate(surrogateDensities):
        assert density._model.likelihood.tempering == temperingSequence2[i], (
            f"Tempering for surrogate {i} did not update correctly. "
            f"Expected {temperingSequence2[i]}, got {density._model.likelihood.tempering}."
        )

    # Edge case: mismatched tempering sequence length
    with pytest.raises(ValueError, match="Tempering only makes sense if the target represents an .*posterior"):
        invalidSequence = [0.1] * (nSurrogates - 1)  # Too short
        mlda.set_tempering_sequence(invalidSequence)

    # Edge case: invalid surrogate densities
    with pytest.raises(ValueError, match="Tempering only makes sense if the target represents an .*posterior"):
        mlda._finestTarget = "InvalidTarget"  # Invalidate the target type
        mlda.set_tempering_sequence(temperingSequence1)
