import pandas as pd
import numpy as np

from vclog import Logger
from probcolli import CGP
from probcolli.entities import CGPInfo


def main():
    x = pd.read_csv('tests/data/x.csv').to_numpy()
    y = pd.read_csv('tests/data/y.csv').to_numpy().flatten()

    x_train = x[:8000, :]
    y_train = y[:8000]

    x_test = x[8000:, :]
    y_test = y[8000:]

    collision_checker = CGP(128, 12)

    collision_checker.train(x_train, y_train, epochs=20)

    info: CGPInfo = collision_checker.predict(x_test)

    success_rate: float = np.sum(np.logical_and(info.decision, y_test))/len(y_test)

    Logger.info(f'{success_rate=:.2f}')

    collision_checker.save('tests/models/cgp/')

    Logger.info('model saved')

    new_cgp: CGP = CGP.from_model('tests/models/cgp/')

    info: CGPInfo = new_cgp.predict(x_test)

    success_rate: float = np.sum(np.logical_and(info.decision, y_test))/len(y_test)

    Logger.info(f'{success_rate=:.2f}')
    Logger.info(f'min: {min(info.decision)}, max: {max(info.decision)}')

    # Try to predict only one value
    value: np.ndarray = np.random.uniform(-1.0, 1.0, size=12)
    info: CGPInfo = new_cgp.predict(value)

    Logger.info('decision: ', info.decision[0])


if __name__ == '__main__':
    main()
