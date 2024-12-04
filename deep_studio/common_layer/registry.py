class BaseRegistry(type):
    """
    BaseRegistry 클래스는 클래스를 등록하고, 이름을 통해 해당 클래스를 인스턴스화할 수 있는 기능을 제공합니다.
    """

    all_registry = dict()

    def __new__(cls, name, bases, attrs):
        new_cls = type.__new__(cls, name, bases, attrs)

        # 각 클래스의 고유한 REGISTRY 생성
        if not hasattr(new_cls, "REGISTRY"):
            new_cls.REGISTRY = {}

        # register 메서드를 각 클래스에 바인딩
        def register(cls, tgt):
            cls.REGISTRY[tgt.__name__] = tgt
            return tgt

        def build(cls, **kwargs):
            name = kwargs.get("name")
            assert name in cls.REGISTRY, f"{name} is not registered."
            return cls.REGISTRY[name](**kwargs)

        new_cls.register = classmethod(register)
        new_cls.build = classmethod(build)

        cls.all_registry[name] = new_cls

        return new_cls


def REGISTRY_FACTORY(name):
    """
    이름을 통해 새로운 레지스트리 클래스를 생성하여 반환합니다.
    """
    # 동적으로 이름이 name인 클래스를 생성하고, BaseRegistry를 메타클래스로 지정
    return BaseRegistry(name, (object,), {})
