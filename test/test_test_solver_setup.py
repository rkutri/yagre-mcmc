import pytest
import numpy as np
import test.testSetup as ts

from inference.data import Data
from model.forwardModel import EvaluationStatus

# Sample configuration
config = {
    'T': 10,
    'alpha': 0.1,
    'gamma': 0.1,
    'nData': 10,
    'dataDim': 2
}

# Sample initial conditions
design = np.random.rand(config['nData'], config['dataDim'])

# Sample parameters
coefficients = np.array([0.2, 0.3])
parameter = ts.LotkaVolterraParameter.from_coefficient(coefficients)


def test_initialize_solver():

    solver = ts.LotkaVolterraSolver(design, config)

    assert solver.x_.shape == design.shape
    assert solver.tBoundary_ == (0., config['T'])
    assert solver.fixedParam_ == [config['alpha'], config['gamma']]
    assert solver.dataShape_ == (config['nData'], config['dataDim'])
    assert solver.param_ == [None, None]
    assert solver.evaluation_ is None
    assert solver.status_ == EvaluationStatus.NONE


def test_interpolate():

    solver = ts.LotkaVolterraSolver(design, config)
    solver.interpolate(parameter)

    np.testing.assert_allclose(solver.param_, np.exp([0.2, 0.3]),
                               rtol=1e-5, atol=1e-8)


def test_invoke():

    solver = ts.LotkaVolterraSolver(design, config)
    solver.interpolate(parameter)
    solver.invoke()

    assert solver.status_ == EvaluationStatus.SUCCESS
    assert solver.evaluation_.shape == (config['nData'], config['dataDim'])


def test_full_solution():

    solver = ts.LotkaVolterraSolver(design, config)
    y0 = np.array([1.0, 1.0])
    t, y = solver.full_solution(parameter, y0)

    assert len(t) > 0
    assert y.shape[1] == len(t)


def test_generate_synthetic_data():

    noise_var = 0.1
    data = ts.generate_synthetic_data(
        parameter, ts.LotkaVolterraSolver(design, config), noise_var)

    assert isinstance(data, Data)
    assert data.size == config['nData']
    assert data.array[0, :].size == config['dataDim']


# Run the tests
if __name__ == "__main__":
    pytest.main()
