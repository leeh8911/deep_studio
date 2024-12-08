import pytest


from deep_studio.common_layer.registry import REGISTRY_FACTORY, RegistryError


def test_basic_register_check():
    """Register의 기본 기능에 대해 확인하는 테스트"""

    TEST_REGISTRY = REGISTRY_FACTORY("TEST_REGISTRY")

    @TEST_REGISTRY.register
    class Foo:

        def __init__(self, name, spiderman, ironman):
            self.name = name
            self.spiderman = spiderman
            self.ironman = ironman

        def do_something(self):
            self.logger.info("this is Foo class")

    foo = TEST_REGISTRY.build(
        **{"name": "Foo", "spiderman": "Tom", "ironman": "Robert"}
    )

    foo.do_something()

    assert foo.name == "Foo"
    assert foo.spiderman == "Tom"
    assert foo.ironman == "Robert"


def test_all_registry_need_to_be_independently():
    """두 개 이상의 Register가 있을 때, REGISTRY는 독립적인지를 테스트"""

    FIRST_REGISTRY = REGISTRY_FACTORY("FIRST_REGISTRY")
    SECOND_REGISTRY = REGISTRY_FACTORY("SECOND_REGISTRY")

    @FIRST_REGISTRY.register
    class Foo:

        def __init__(self, name, spiderman, ironman):
            self.name = name

            self.spiderman = spiderman
            self.ironman = ironman

        def do_something(self):
            self.logger.info("this is Foo class")

    assert FIRST_REGISTRY.build(
        **{"name": "Foo", "spiderman": "Tom", "ironman": "Robert"}
    )

    with pytest.raises(RegistryError):
        SECOND_REGISTRY.build(
            **{"name": "Foo", "spiderman": "Tom", "ironman": "Robert"}
        )
