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

        def __init__(self, spiderman, ironman):
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

        def __init__(self, spiderman, ironman):
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


def test_dynamically_load_from_ext_module():
    """외부 모듈(여기선 torch)에서 등록되지 않은 클래스를 사용"""

    LOSS_REGISTRY = make_registry("LOSS_REGISTRY")

    criterion = LOSS_REGISTRY.build(**{"name": "torch.nn.L1Loss"})
    assert criterion.__class__.__name__ == "torch.nn.L1Loss_Logger"
    # 'torch.nn.'이 추가로 붙은 이유는 REGISTRY.build 사용 시 입력되는 이름을 동일하게 유지하기 위해서. torch.nn.L1Loss를 그대로 사용하면 criterion.__class__.__name__은 L1Loss가 됨
    # 이름 뒤에 _Logger가 붙은 이유는 builder로 생성하는 모든 클래스에는 self.logger가 추가되기 때문 (Mixin을 통해 새로운 클래스를 만든 다음 이름 뒤에 Logger를 붙임)
