from karoo_gp import Terminal, Terminals

def test_terminal():
    t = Terminal('a')
    assert t.symbol == 'a'
    assert t.type == float
    assert repr(t) == "<Terminal: symbol='a' type='float'>"

    t = Terminal(1, int)
    assert t.symbol == 1
    assert t.type == int

def test_terminals():
    ts = Terminals(['a', 'b'], constants=[1, 2])
    assert ts.variables['a'].symbol == 'a'
    assert ts.constants[0].symbol == 1
    assert len(ts.get()) == 4
