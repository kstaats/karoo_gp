from karoo_gp import NodeData, function_lib, get_nodes, get_function_node

def test_node_lib_data():
    t = NodeData('a', 'terminal')
    assert t.label == 'a'
    assert t.node_type == 'terminal'
    assert t.arity == 0
    assert t.min_depth == 0
    assert t.child_type == None

def test_node_lib_get():
    types = ['operator', 'bool', 'cond']
    for t in types:
        nodes = get_nodes(t)
        assert len(nodes) > 0
        assert all([n.node_type == t for n in nodes])

def test_node_lib_get_function():
    for f in function_lib:
        _f = get_function_node(f.label)
        assert repr(f) == repr(_f)
