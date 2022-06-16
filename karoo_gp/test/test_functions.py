from karoo_gp import Function, Functions

def test_function():
    func = Function('+')
    assert func.symbol == '+'
    assert func.type == 'arithmetic'
    assert func.arity == 2
    assert repr(func) == "<Function: symbol=+, type='arithmetic', arity=2>"

def test_functions():
    funcs = Functions(['+', '-'])
    assert len(funcs.get()) == 2
    assert funcs.functions[0].symbol == '+'
    assert repr(funcs) == "<Functions: 2(+-)>"

def test_functions_arithmetic():
    a_funcs = Functions.arithmetic()
    assert len(a_funcs.get()) == 5

def test_functions_logic():
    l_funcs = Functions.logic()
    assert len(l_funcs.get()) == 10

def test_functions_math():
    m_funcs = Functions.math()
    assert len(m_funcs.get()) == 13
