"""
"""

from deep_studio.exp_layer.experiment import Experiment


def main():

    exp = Experiment()

    exp.build()

    exp.run()

    exp.visualization()


if __name__ == "__main__":
    main()
