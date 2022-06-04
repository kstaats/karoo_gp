import numpy as np

function_dict = {'+': dict(func_type='arithmetic', arity=2),
                 '-': dict(func_type='arithmetic', arity=2),
                 '*': dict(func_type='arithmetic', arity=2),
                 '/': dict(func_type='arithmetic', arity=2),
                 '**': dict(func_type='arithmetic', arity=2),
                 'neg': dict(func_type='logic', arity=1),
                 'and': dict(func_type='logic', arity=2),
                 'or': dict(func_type='logic', arity=2),
                 'not': dict(func_type='logic', arity=1),
                 '==': dict(func_type='logic', arity=2),
                 '!=': dict(func_type='logic', arity=2),
                 '<': dict(func_type='logic', arity=2),
                 '<=': dict(func_type='logic', arity=2),
                 '>': dict(func_type='logic', arity=2),
                 '>=': dict(func_type='logic', arity=2),
                 'abs': dict(func_type='math', arity=1),
                 'sign': dict(func_type='math', arity=1),
                 'square': dict(func_type='math', arity=1),
                 'sqrt': dict(func_type='math', arity=1),
                 'pow': dict(func_type='math', arity=1),
                 'log': dict(func_type='math', arity=1),
                 'log1p': dict(func_type='math', arity=1),
                 'cos': dict(func_type='math', arity=1),
                 'sin': dict(func_type='math', arity=1),
                 'tan': dict(func_type='math', arity=1),
                 'acos': dict(func_type='math', arity=1),
                 'asin': dict(func_type='math', arity=1),
                 'atan': dict(func_type='math', arity=1)}

class Function:
    """A branch node containing an function"""
    def __init__(self, symbol):
        if symbol not in function_dict:
            raise ValueError("Unrecognized function key:", symbol)
        self.symbol = symbol
        self.func_type = function_dict[symbol]['func_type']
        self.arity = function_dict[symbol]['arity']

    def __repr__(self):
        return f"<Function: {self.symbol}({self.func_type})>"

class Functions():
    def __init__(self, function_list):
        self.functions = [Function(f) for f in function_list]

    def __repr__(self):
        f_string = "".join([f.symbol for f in self.functions])
        return f"<Functions: {len(self.functions)}({f_string[:10]})>"

    def get(self):
        return np.array(self.functions)

    @classmethod
    def load(cls, func_types):
        funcs = []
        for k, v in function_dict.items():
            if v['func_type'] in func_types:
                funcs.append(k)
        return cls(funcs)

    @classmethod
    def arithmetic(cls): return cls.load(['arithmetic'])
    @classmethod
    def logic(cls): return cls.load(['logic'])
    @classmethod
    def math(cls): return cls.load(['math'])

