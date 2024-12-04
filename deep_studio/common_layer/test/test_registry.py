import pytest

from deep_studio.common_layer.registry import REGISTRY_FACTORY


def test_registry():
    TEST_REGISTRY = REGISTRY_FACTORY("TEST_REGISTRY")

    @TEST_REGISTRY.register
    class Foo:
        def __init__(self, name, spiderman, ironman):
            self.name = name
            self.spiderman = spiderman
            self.ironman = ironman

    foo = TEST_REGISTRY.build(
        **{"name": "Foo", "spiderman": "Tom", "ironman": "Robert"}
    )

    assert foo.name == "Foo"
    assert foo.spiderman == "Tom"
    assert foo.ironman == "Robert"
