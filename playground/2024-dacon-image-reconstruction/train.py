""" 2024.Dacon.ImageReconstruction challenge를 학습하기 위한 러너
"""

from deep_studio.exp_layer.experiment import Experiment


def main():

    exp = Experiment()

    exp.build()

    exp.run()

    exp.visualization()


if __name__ == "__main__":
    main()
