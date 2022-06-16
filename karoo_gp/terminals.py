from itertools import zip_longest
import numpy as np

# TODO: Doesn't need to be a class
class Terminal:
    """A branch node containing a variable or constant"""
    def __init__(self, symbol, type=float):
        self.symbol = symbol
        # TODO: Currently the type isn't used, and the engine uses float32 by
        # default. The type attribute here could theoretically be used to
        # support boolean or categorical values for terminals. Currently the
        # user has to remove or convert these to numeric in preprocessing.
        self.type = type

    def __repr__(self):
        return (f"<Terminal: symbol={self.symbol!r} "
                f"type={self.type.__name__!r})>")

class Terminals:
    def __init__(self, variables, constants=None, types=None,
                 default_type=float):
        """Return a Terminals object to store active terminals"""
        self.variables = {}
        constants = [] if constants is None else constants
        types = [] if types is None else types
        for s, t in zip_longest(variables, types, fillvalue=default_type):
            if s in self.variables:
                raise ValueError("Terminals must be unique:", s)
            self.variables[s] = Terminal(s, t)
        self.constants = None if constants is None else [
                          Terminal(c, type(c)) for c in constants]

    def __repr__(self):
        v_string = "".join(self.variables)
        c_string = "".join([str(c.symbol) for c in self.constants])
        t_string = f"{v_string}, {c_string}"
        return f"<Terminals: {len(self.get())}({t_string[:10]})>"

    def get(self):
        return np.array(list(self.variables.values()) + self.constants)
