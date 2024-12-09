"""registry.py

사용자가 정의한 클래스를 등록하여 손쉽게 다룰 수 있게 도와주기 위한 레지스터
"""

import logging
from typing import Type, Dict, Any


class RegistryError(Exception):
    """Registry에 관한 오류"""


class BaseRegistry(type):
    """
    BaseRegistry 클래스는 클래스를 등록하고, 이름을 통해 해당 클래스를 인스턴스화할 수 있는 기능을 제공합니다.
    """

    all_registry = dict()

    REGISTRY: Dict[str, Type[Any]]
    register: classmethod
    build: classmethod

    def __new__(mcs, name, bases, attrs):
        new_cls = type.__new__(mcs, name, bases, attrs)

        # 각 클래스의 고유한 REGISTRY 생성
        if not hasattr(new_cls, "REGISTRY"):
            new_cls.REGISTRY = {}

        # register 메서드를 각 클래스에 바인딩
        def register(cls, tgt):
            cls.REGISTRY[tgt.__name__] = tgt
            return tgt

        def build(cls, **kwargs):
            name = kwargs.get("name")
            if name not in cls.REGISTRY:
                raise RegistryError(
                    f"Class '{name}' is not registered in {cls.__name__}."
                )

            ret = cls.REGISTRY[name](**kwargs)
            ret.__setattr__("logger", cls.logger.getChild(name))
            return ret

        new_cls.register = classmethod(register)
        new_cls.build = classmethod(build)

        mcs.all_registry[name] = new_cls

        return new_cls


def REGISTRY_FACTORY(name, log_level: str = "info"):  # pylint: disable=invalid-name
    """
    이름을 통해 새로운 레지스트리 클래스를 생성하여 반환합니다.
    """

    logger = logging.getLogger(name=name)
    logger.setLevel(log_level.upper())

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(pathname)s:%(lineno)d - %(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return BaseRegistry(name, (object,), {"logger": logger})
