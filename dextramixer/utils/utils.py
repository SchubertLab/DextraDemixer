
#Metaclass for Plugins
class RegisteredModel(type):
    def __new__(cls, clsname, superclasses, attributedict):
        newclass = type.__new__(cls, clsname, superclasses, attributedict)
        if not hasattr(cls, 'registry'):
            cls.registry = dict()

        # condition to prevent base class registration
        if superclasses:
            cls.registry.setdefault(str(newclass().name).lower(), newclass)
        return newclass
