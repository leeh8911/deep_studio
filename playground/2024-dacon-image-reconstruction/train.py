""" 2024.Dacon.ImageReconstruction challenge를 학습하기 위한 러너
"""

import sys
from pathlib import Path

from torchvision.transforms.functional import to_pil_image

# 프로젝트 경로 설정
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from models.model_interface import Dacon2024ImageReconstructionModelInterface
from dataset import Dacon2024ImageReconstructionDataset

from deep_studio.exp_layer.experiment import Experiment


def main():
    exp = Experiment()

    exp.train()

    res = exp.test()

    for idx, image in enumerate(res):
        image = to_pil_image(image)

        image.save(
            f"playground/2024-dacon-image-reconstruction/output/TEST_{str(idx).zfill(3)}.png"
        )


if __name__ == "__main__":
    main()
