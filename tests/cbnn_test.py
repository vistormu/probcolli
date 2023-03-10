import pandas as pd
import numpy as np

from vclog import Logger
from probcolli import CBNN
from probcolli.entities import CBNNInfo


def main():
    x = pd.read_csv('tests/data/x.csv').to_numpy()
    y = pd.read_csv('tests/data/y.csv').to_numpy().flatten()

    x_train = x[:8000, :]
    y_train = y[:8000]

    x_test = x[8000:, :]
    y_test = y[8000:]

    cbnn = CBNN(12)

    cbnn.train(x_train, y_train)

    info: CBNNInfo = cbnn.predict(x_test)

    success_rate: float = np.sum(np.logical_and(info.decision, y_test))/len(y_test)

    Logger.info(f'{success_rate=:.2f}')

    cbnn.save('tests/models/cgp/')

    Logger.info('model saved')

    new_cbnn: CBNN = CBNN.from_model('tests/models/cgp/')

    info = new_cbnn.predict(x_test)

    success_rate: float = np.sum(np.logical_and(info.decision, y_test))/len(y_test)

    Logger.info(f'{success_rate=:.2f}')

    # Try to predict only one value
    value: np.ndarray = np.random.uniform(-1.0, 1.0, size=12)
    info: CBNNInfo = new_cbnn.predict(value)

    Logger.info('decision: ', info.decision[0])


if __name__ == '__main__':
    main()
