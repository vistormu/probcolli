import pandas as pd
import numpy as np

from probcolli import CBNN, Logger
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

    new_cbnn: CBNN = CBNN.load('tests/models/cgp/')

    info  = new_cbnn.predict(x_test)

    success_rate: float = np.sum(np.logical_and(info.decision, y_test))/len(y_test)

    Logger.info(f'{success_rate=:.2f}')


if __name__ == '__main__':
    main()
