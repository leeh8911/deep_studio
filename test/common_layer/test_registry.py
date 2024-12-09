"""
test_registry.py

Registry에 대해 테스트하는 모듈입니다.
"""

import pytest


from deep_studio.common_layer.registry import make_registry, RegistryError


def test_basic_register_check():
    """Register의 기본 기능에 대해 확인하는 테스트"""

    TEST_REGISTRY = make_registry("TEST_REGISTRY")

    @TEST_REGISTRY.register
    class Marvel:
        """테스트를 위한 임시 클래스"""

        def __init__(self, name, spiderman, ironman):
            self.name = name
            self.spiderman = spiderman
            self.ironman = ironman

        def do_something(self):
            """테스트를 위한 임시 함수"""
            self.logger.info("this is Marvel class")

    marvel: Marvel = TEST_REGISTRY.build(
        **{"name": "Marvel", "spiderman": "Tom", "ironman": "Robert"}
    )

    marvel.do_something()

    assert marvel.name == "Marvel"
    assert marvel.spiderman == "Tom"
    assert marvel.ironman == "Robert"


def test_all_registry_need_to_be_independently():
    """두 개 이상의 Register가 있을 때, REGISTRY는 독립적인지를 테스트"""

    FIRST_REGISTRY = make_registry("FIRST_REGISTRY")
    SECOND_REGISTRY = make_registry("SECOND_REGISTRY")

    @FIRST_REGISTRY.register
    class Marvel:  # pylint: disable=<unused-variable>
        """테스트를 위한 임시 클래스"""

        def __init__(self, name, spiderman, ironman):
            self.name = name

            self.spiderman = spiderman
            self.ironman = ironman

        def do_something(self):
            """테스트를 위한 임시 함수"""
            self.logger.debug("this is Marvel class")

    assert FIRST_REGISTRY.build(
        **{"name": "Marvel", "spiderman": "Tom", "ironman": "Robert"}
    )

    with pytest.raises(RegistryError):
        SECOND_REGISTRY.build(
            **{"name": "Marvel", "spiderman": "Tom", "ironman": "Robert"}
        )
