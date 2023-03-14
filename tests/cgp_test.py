import pandas as pd
import numpy as np
import bgplot as bgp

from bgplot.entities import Point
from vclog import Logger
from probcolli import CGP
from probcolli.entities import CGPInfo


def main():
    # Read data
    points: np.ndarray = pd.read_csv('tests/data/end_effector.csv').to_numpy()
    points = points/np.max(points)
    labels: np.ndarray = pd.read_csv('tests/data/labels.csv').to_numpy().flatten()

    # Hyperparameters
    inducing_points: int = 512
    dof: int = 3
    epochs: int = 10
    batch_size: int = 128
    variational_lr: float = 0.5
    hyperparameter_lr: float = 0.1
    beta: float = 0.1

    # Train
    cgp: CGP = CGP(inducing_points, dof)
    cgp.train(points, labels, epochs, batch_size, variational_lr, hyperparameter_lr)

    for name, param in cgp.model.named_parameters():
        if param.requires_grad:
            Logger.debug(name, param.data, sep=' ')

    # Predict
    info: CGPInfo = cgp.predict(points, beta)

    # Visualize
    figure: bgp.Graphics = bgp.Graphics()
    figure.set_limits(xlim=(0.0, 0.5), ylim=(0, 0.5), zlim=(0.0, 1.2))
    figure.set_view(180.0, 20.0)
    figure.disable('grid', 'ticks', 'axes', 'walls')
    figure.set_background_color(bgp.Colors.white)

    points_q5: list[Point] = [Point(*point) for point, decision in zip(points, info.decision) if decision <= 0.25]
    points_q4: list[Point] = [Point(*point) for point, decision in zip(points, info.decision) if decision <= 0.5 and decision > 0.25]
    points_q3: list[Point] = [Point(*point) for point, decision in zip(points, info.decision) if decision <= 0.75 and decision > 0.5]
    points_q2: list[Point] = [Point(*point) for point, decision in zip(points, info.decision) if decision <= 0.9 and decision > 0.75]
    points_q1: list[Point] = [Point(*point) for point, decision in zip(points, info.decision) if decision >= 0.9]

    collided_points: list[Point] = [Point(*point) for point, label in zip(points, labels) if label]

    figure.add_points(points_q5, style=',', color=bgp.Colors.blue)
    figure.add_points(points_q4, style=',', color=bgp.Colors.green)
    figure.add_points(points_q3, style=',', color=bgp.Colors.yellow)
    figure.add_points(points_q2, style=',', color=bgp.Colors.yellow_orange)
    figure.add_points(points_q1, style=',', color=bgp.Colors.red)

    figure.add_points(collided_points, style=',')

    figure.show()


if __name__ == '__main__':
    main()
