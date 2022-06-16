import numpy as np

function_dict = {
    '+': dict(type='arithmetic', arity=2),
    '-': dict(type='arithmetic', arity=2),
    '*': dict(type='arithmetic', arity=2),
    '/': dict(type='arithmetic', arity=2),
    '**': dict(type='arithmetic', arity=2),
    'neg': dict(type='logic', arity=1),
    'and': dict(type='logic', arity=2),
    'or': dict(type='logic', arity=2),
    'not': dict(type='logic', arity=1),
    '==': dict(type='logic', arity=2),
    '!=': dict(type='logic', arity=2),
    '<': dict(type='logic', arity=2),
    '<=': dict(type='logic', arity=2),
    '>': dict(type='logic', arity=2),
    '>=': dict(type='logic', arity=2),
    'abs': dict(type='math', arity=1),
    'sign': dict(type='math', arity=1),
    'square': dict(type='math', arity=1),
    'sqrt': dict(type='math', arity=1),
    'pow': dict(type='math', arity=1),
    'log': dict(type='math', arity=1),
    'log1p': dict(type='math', arity=1),
    'cos': dict(type='math', arity=1),
    'sin': dict(type='math', arity=1),
    'tan': dict(type='math', arity=1),
    'acos': dict(type='math', arity=1),
    'asin': dict(type='math', arity=1),
    'atan': dict(type='math', arity=1),
}

# TODO: Doesn't need to be a class
class Function:
    """A branch node containing a function"""
    def __init__(self, symbol):
        if symbol not in function_dict:
            raise ValueError("Unrecognized function key:", symbol)
        self.symbol = symbol
        self.type = function_dict[symbol]['type']
        self.arity = function_dict[symbol]['arity']

    def __repr__(self):
        return (f'<Function: symbol={self.symbol}, type={self.type!r}, '
                f'arity={self.arity}>')

class Functions:
    def __init__(self, function_list):
        self.functions = [Function(f) for f in function_list]

    @classmethod
    def load(cls, types):
        funcs = []
        for k, v in function_dict.items():
            if v['type'] in types:
                funcs.append(k)
        return cls(funcs)

    # Find a better solution for this
    @classmethod
    def arithmetic(cls):
        return cls.load(['arithmetic'])

    @classmethod
    def logic(cls):
        return cls.load(['logic'])

    @classmethod
    def math(cls):
        return cls.load(['math'])

    def __repr__(self):
        f_string = "".join([f.symbol for f in self.functions])
        return f"<Functions: {len(self.functions)}({f_string[:10]})>"

    def get(self):
        return np.array(self.functions)
