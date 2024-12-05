import torch
from abc import abstractmethod
from typing import Any


class BaseModelInterface(torch.nn.Module):
    """BaseModelInterface는 실제 실험에 사용될 모델 구현에 앞서 정의해야 하는 인터페이스 클래스입니다.
    이 클래스를 상속받은 ModelInterface를 먼저 구현한 뒤, 실제로 실험에서 사용할 모델을 구현합니다.

    example:
    >>> class ModelInterface(BaseModelInterface):
    >>>     # 여기서는 'forward_train'과 'forward_test'만 구현하고, 'forward'는 실제 구현할 함수에서 정의합니다.
    >>>     (생략)
    >>>     def forward_train(self, *arg, **kwargs): # 필요에 따라 입/출력 인자를 실험 설계에 맞춰 추가하거나 변경합니다.
    >>>         pass
    >>>     def forward_test(self, *arg, **kwargs): # 필요에 따라 입/출력 인자를 실험 설계에 맞춰 추가하거나 변경합니다.
    >>>         pass
    >>> class ConcreteModel(ModelInterface):
    >>>     (생략)
    >>>     def forward(self, *arg, **kwargs): # 필요에 따라 입/출력 인자를 실험 설계에 맞춰 추가하거나 변경합니다.
    """

    @abstractmethod
    def forward_train(self, *args, **kwargs) -> Any:
        """모델에 데이터를 넣어 예측값과 손실값, 평가지표를 출력하기 위한 추상 함수

        Raises:
            NotImplementedError: _description_

        Returns:
            Any: 필요에 따라 예측값과 손실값, 평가지표를 출력함
        """
        raise NotImplementedError("You need implement this 'forward_train'")

    @abstractmethod
    def forward_test(self, *args, **kwargs) -> Any:
        """모델에 데이터를 넣어 예측값을 출력하기 위한 추상 함수

        Raises:
            NotImplementedError: _description_

        Returns:
            Any: 모델에서 필요한 예측값만 출력함
        """
        raise NotImplementedError("You need implement this 'forward_test'")
