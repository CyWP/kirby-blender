class Opt:
    
    def __init__(self, initial_dict):

        self._data = initial_dict

    def __getattr__(self, name):

        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'Opt' object has no attribute '{name}'")

    def __setattr__(self, name, value):

        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __delattr__(self, name):

        if name in self._data:
            del self._data[name]
        else:
            raise AttributeError(f"'Opt' object has no attribute '{name}'")

    def to_dict(self):

        return self._data.copy()