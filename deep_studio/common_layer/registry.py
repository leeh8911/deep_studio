"""registry.py

사용자가 정의한 클래스를 등록하여 손쉽게 다룰 수 있게 도와주기 위한 레지스터
"""

import logging
from typing import Type, Dict, Any


class RegistryError(Exception):
    """Registry에 관한 오류"""


class LogMixin:
    """Logging을 위한 Mixin 클래스.
    Registry.build를 이용해서 생성할 수 있다.
    """

    def __init__(self):
        self.__logger = None

    @property
    def logger(self) -> logging.Logger:
        """logger

        Returns:
            _type_: _description_
        """
        return self.__logger

    @logger.setter
    def logger(self, logger: logging.Logger):
        self.__logger = logger


class BaseRegistry(metaclass=type):
    """
    BaseRegistry 클래스는 클래스를 등록하고, 이름을 통해 해당 클래스를 인스턴스화할 수 있는 기능을 제공합니다.
    """

    all_registry: Dict[str, Type["BaseRegistry"]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        각 서브클래스의 고유한 REGISTRY와 logger를 생성 및 등록합니다.
        """
        super().__init_subclass__(**kwargs)
        cls.REGISTRY: Dict[str, Type[Any]] = {}
        cls.logger = logging.getLogger(cls.__name__)  # 각 클래스의 이름으로 로거 설정
        BaseRegistry.all_registry[cls.__name__] = cls

    @classmethod
    def register(cls, tgt: Type[Any]) -> Type[Any]:
        """
        클래스를 REGISTRY에 등록합니다.
        """
        cls.REGISTRY[tgt.__name__] = tgt
        cls.logger.debug(
            "Class '%s' registered in '%s'.",
            tgt.__name__,
            cls.__name__,
        )
        return tgt

    @classmethod
    def build(cls, **kwargs: Any) -> Any:
        """
        등록된 클래스를 이름을 통해 인스턴스화합니다.
        """
        name = kwargs.get("name")
        if name not in cls.REGISTRY:
            raise RegistryError(f"Class '{name}' is not registered in {cls.__name__}.")

        new_type = type(f"{name}_Logger", (cls.REGISTRY[name], LogMixin), {})

        instance = new_type(**kwargs)
        instance.logger = cls.logger.getChild(new_type.__name__)
        cls.logger.debug(
            "Instance of '%s' created in '%s'.", new_type.__name__, cls.__name__
        )
        return instance


def make_registry(name: str, log_level: str = "info") -> Type[BaseRegistry]:
    """
    이름을 통해 새로운 레지스트리 클래스를 생성하여 반환합니다.
    """

    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return type(name, (BaseRegistry,), {"logger": logger})
