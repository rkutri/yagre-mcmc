from yagremcmc.utility.boilerplate import create_logger
from yagremcmc.chain.diagnostics import AcceptanceRateDiagnostics


class VerbosityLevel:

    OFF = 0
    FULL = 1


class VerbosityController:

    def __init__(self, nPrintIntervals=20, minInterval=10):

        self._verbosityLevel = VerbosityLevel.FULL

        self._nPrintIntervals = nPrintIntervals
        self._minInterval = minInterval

        self._logger = create_logger(f"MH_{id(self)}")

        self._printInterval = None
        self._diagnostics = None

    def prepare(self, chainLength, diagnostics):

        self._printInterval = max(chainLength // self._nPrintIntervals,
                                  self._minInterval)

        if isinstance(diagnostics, AcceptanceRateDiagnostics):
            diagnostics.lag = self._printInterval

        self._diagnostics = diagnostics

    def turn_off(self):
        self._verbosityLevel = VerbosityLevel.OFF

    def run(self, iterIdx):

        if self._verbosityLevel == VerbosityLevel.OFF:
            return

        if iterIdx == 0:
            self._logger.info(f"Starting chain.")

        if iterIdx % self._printInterval == 0 and iterIdx > 0:

            self._logger.info(
                f"{iterIdx} steps computed. Calculating diagnostics.")
            self._diagnostics.print_diagnostics(self._logger)
