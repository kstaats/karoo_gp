from itertools import zip_longest
import numpy as np

class Terminal:
    """A branch node containing a variable or constant"""
    def __init__(self, symbol, t_type=float):
        self.symbol = symbol
        self.t_type = t_type

    def __repr__(self):
        return f"<Terminal: {self.symbol}({str(self.t_type)})>"

class Terminals:
    def __init__(self, variables, types=None, constants=[], default_type=float):
        self.variables = {}
        types = types or []
        for s, t in zip_longest(variables, types):
            if s in self.variables:
                raise ValueError("Terminals must be unique:", s)
            t_type = t or default_type
            self.variables[s] = Terminal(s, t_type)
        self.constants = [Terminal(c, type(c)) for c in constants]
        # TODO: Add class mappings

    def __repr__(self):
        v_string = "".join(self.variables)
        c_string = "".join([str(c.symbol) for c in self.constants])
        t_string = f"{v_string}, {c_string}"
        return f"<Terminals: {len(self.get())}({t_string[:10]})>"

    def get(self):
        return np.array(list(self.variables.values()) + self.constants)
