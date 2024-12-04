# Deep Studio Project

## Overview

**Deep Studio**는 딥러닝 프로젝트의 효율적이고 체계적인 관리를 위한 통합 프레임워크입니다. 이 프로젝트는 데이터 전처리, 모델 개발, 실험, 배포의 모든 단계를 다루며, 특히 다양한 연구와 실험의 목적을 달성하기 위한 확장성과 모듈화된 구조를 갖추고 있습니다.

본 프로젝트는 다음과 같은 주요 디렉토리 구조로 이루어져 있습니다:

```
deep_studio/
├── common_layer/        # 공용 타입 및 유틸리티 정의
├── data_layer/          # 데이터 관련 로직 및 전처리 모듈
├── deploy_layer/        # 모델 배포 관련 모듈
├── exp_layer/           # 실험 관리 및 구성 관련 모듈
├── model_layer/         # 모델 관련 모듈 (SOTA 모델 및 공용 모델)
│   └── model/           # SOTA 모델 정의 및 공용 모델 유틸리티
└── playground/          # 실험과 프로젝트를 위한 공간
    └── 2024-dacon-image-reconstruction/   # 특정 실험 프로젝트 예제
```

## Directory Details

### `common_layer/`

- **Purpose**: 프로젝트 전체에서 공통으로 사용되는 타입(`Loss`, `Image`, `Metrics`) 및 유틸리티를 정의합니다.
- **Key Files**:
  - `types.py`: 공용 타입 정의 파일로, 모델이나 데이터 관련 작업에서 일관성을 유지하기 위해 사용됩니다.

### `data_layer/`

- **Purpose**: 데이터 관련 작업을 수행하는 모듈들이 위치합니다. 데이터 수집, 전처리, 로더 기능 등을 제공합니다.
- **Key Files**:
  - `__init__.py`: 데이터 레이어 초기화 파일. 필요 시 데이터 전처리 파이프라인 초기화를 포함할 수 있습니다.

### `deploy_layer/`

- **Purpose**: 훈련된 모델을 배포하는 데 필요한 기능들을 정의합니다. 예를 들어, 모델 서빙을 위한 API 생성 기능을 포함할 수 있습니다.

### `exp_layer/`

- **Purpose**: 실험과 관련된 구성 요소들을 관리합니다. 실험의 구성(`config`) 및 관리 로직이 위치하며, 각 실험의 설정을 효율적으로 관리할 수 있도록 도와줍니다.

### `model_layer/`

- **Purpose**: 모델 정의 및 학습 관련 모듈들이 위치합니다.
  - **`model/`**: 주로 SOTA(State-of-the-art) 연구에서 제안된 모델이나 기존 실험에서 사용된 모델을 모아두기 위한 디렉토리입니다. 공용으로 재사용할 수 있는 모델을 정의하여 다른 실험에서 손쉽게 가져다 사용할 수 있습니다.
- **Key Files**:
  - `model_interface.py`: 모델 인터페이스 정의 파일로, 모든 모델이 따라야 할 기본 구조를 정의합니다.
  - `model/`: 각종 SOTA 모델의 구현체가 위치하는 디렉토리입니다.

### `playground/`

- **Purpose**: 실험적 작업이나 특정 프로젝트를 위한 공간입니다. 각 프로젝트는 독립적으로 진행되며, 프로젝트별로 모델을 자유롭게 정의하고 테스트할 수 있습니다.
  - **`2024-dacon-image-reconstruction/`**: 특정 실험 프로젝트 예제로, 데이터셋 준비, 실험 구성 파일(`exp-config.py`), 학습 스크립트(`train.py`) 등이 포함되어 있습니다.
  - **`models/`**: 해당 실험 프로젝트에 특화된 커스터마이징 모델들이 정의되어 있습니다. 실험의 요구사항에 맞춰 모델을 설계하고 수정할 수 있습니다.

## How to Use

1. **환경 설정**:
   - Python 환경에서 필요한 패키지를 설치합니다. `Poetry`를 사용하여 의존성을 관리합니다.

   ```bash
   poetry install
   ```

2. **실험 프로젝트 실행**:
   - `playground` 디렉토리에서 각 실험 프로젝트를 실행할 수 있습니다. 예를 들어, `2024-dacon-image-reconstruction` 프로젝트에서 학습을 실행하려면:

   ```bash
   cd playground/2024-dacon-image-reconstruction
   python train.py
   ```

3. **모델 사용**:
   - **SOTA 모델**은 `model_layer/model`에서 가져와 사용 가능합니다. 각 모델은 사용 예제와 함께 제공되므로 필요한 경우 해당 모델을 임포트하여 실험에 사용하세요.

   ```python
   from deep_studio.model_layer.model.resnet import ResNet50

   model = ResNet50(pretrained=True)
   ```

4. **실험 구성 및 관리**:
   - `exp_layer`에서 제공하는 모듈을 통해 실험의 구성 파일을 정의하고, 이를 통해 다양한 실험 설정을 자동화할 수 있습니다.

## Contribution Guidelines

- **코드 중복 방지**: 가능한 한 SOTA 모델들은 `model_layer/model`에 모아두고, 각 실험에서는 해당 모델들을 가져다 사용하는 방식으로 코드의 중복을 최소화해주세요.
- **문서화**: 새로운 실험이나 모델을 추가할 때는 관련된 **README 파일**과 **간단한 사용 예제**를 작성하여 다른 개발자들이 쉽게 이해할 수 있도록 합니다.
- **PR 및 코드 리뷰**: 기능 추가나 수정 시 **Pull Request(PR)**를 통해 코드를 제출하고, 팀원들과의 **코드 리뷰**를 통해 개선점을 논의합니다.

## Future Plans

- **모델 등록 시스템 구축**: `Registry` 패턴을 사용해 모델들을 일관된 방식으로 등록하고 참조할 수 있는 시스템을 추가할 예정입니다.
- **문서화 확장**: 프로젝트 내 모든 레이어와 모델 사용법에 대한 자세한 문서화를 강화할 계획입니다.
- **자동화된 실험 관리**: `exp_layer`를 확장하여 실험 설정과 결과 추적을 자동으로 할 수 있는 기능을 추가할 예정입니다.

## License

본 프로젝트는 [MIT License](LICENSE)에 따라 배포됩니다.

## Contact

프로젝트와 관련된 문의사항은 [your_email@example.com](mailto:your_email@example.com)으로 연락주시기 바랍니다.

---

Deep Studio 프로젝트에 관심을 가져주셔서 감사합니다! 😊 앞으로도 더 많은 연구와 실험이 이 프레임워크를 통해 가능해지기를 기대합니다.
