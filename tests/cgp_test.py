import pandas as pd
import numpy as np

from probcolli import CGP, Logger
from probcolli.entities import CGPInfo


def main():
    x = pd.read_csv('tests/data/x.csv').to_numpy()
    y = pd.read_csv('tests/data/y.csv').to_numpy().flatten()

    x_train = x[:8000, :]
    y_train = y[:8000]

    x_test = x[8000:, :]
    y_test = y[8000:]

    collision_checker = CGP(1024, 12)

    collision_checker.train(x_train, y_train, epochs=20)

    info: CGPInfo = collision_checker.predict(x_test)

    success_rate: float = np.sum(np.logical_and(info.decision, y_test))/len(y_test)

    Logger.info(f'{success_rate=:.2f}')


if __name__ == '__main__':
    main()
