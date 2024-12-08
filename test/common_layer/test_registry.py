import pytest


from deep_studio.common_layer.registry import REGISTRY_FACTORY


def test_basic_regist_check():

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

    assert not "Foo" in SECOND_REGISTRY.REGISTRY
    assert "Foo" in FIRST_REGISTRY.REGISTRY
