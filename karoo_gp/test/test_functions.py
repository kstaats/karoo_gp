from karoo_gp import Function, Functions

def test_operator():
    func = Function('+')
    assert func.symbol == '+'
    assert func.func_type == 'arithmetic'
    assert func.arity == 2
    assert repr(func) == '<Function: +(arithmetic)>'

def test_operators():
    funcs = Functions(['+', '-'])
    assert len(funcs.get()) == 2
    assert funcs.functions[0].symbol == '+'

    a_funcs = Functions.arithmetic()
    assert len(a_funcs.get()) == 5

    l_funcs = Functions.logic()
    assert len(l_funcs.get()) == 10

    m_funcs = Functions.math()
    assert len(m_funcs.get()) == 13
