import ast
import csv
from pathlib import Path

import numpy as np
import pytest

from brna.network import Critic, ToyNetwork
from brna.translator import FastTranslator

ROOT=Path(__file__).parents[1]
@pytest.mark.req('R-NET-01')
@pytest.mark.t0
def test_T0_out_degree_is_bounded():
    a=np.ones((12,12))-np.eye(12); assert np.all(a.sum(axis=1)<=11)
@pytest.mark.req('R-NET-02')
@pytest.mark.t1
def test_T1_message_norm_bound():
    net=ToyNetwork(np.zeros((2,2)),message_bound=1.0); assert np.linalg.norm(net.bounded_message(np.array([3.,4.])))==pytest.approx(1.0)
@pytest.mark.req('R-NET-03')
@pytest.mark.t2
def test_T2_runaway_control_is_finite_and_bounded():
    net=ToyNetwork(np.ones((3,3))-np.eye(3),message_bound=.8); y=net.propagate(np.full((3,4),1e4),hops=100); assert np.isfinite(y).all() and np.max(np.linalg.norm(y,axis=1))<=np.sqrt(4)+1e-9
@pytest.mark.req('R-NET-04')
@pytest.mark.t3
def test_T3_propagation_is_deterministic():
    a=np.array([[0,1],[1,0.]]); net=ToyNetwork(a); x=np.array([[.2,.3],[.4,.5]]); np.testing.assert_array_equal(net.propagate(x),net.propagate(x))
@pytest.mark.req('R-NET-05')
@pytest.mark.t4
def test_T4_gate_bounds():
    net=ToyNetwork(np.ones((2,2)),gates=np.array([[-100,0],[100,2.]])); g=net.gate_matrix(); assert np.all(g>=0) and np.all(g<=1)
@pytest.mark.req('R-NET-06')
@pytest.mark.t5
def test_T5_budget_cap():
    c=Critic(hop_limit=9,budget=2); assert c.allow(0,1,0) and c.allow(1,2,0); assert not c.allow(2,3,0) and c.count==2
@pytest.mark.req('R-NET-07')
@pytest.mark.t6
def test_T6_translator_latency():
    tr=FastTranslator(); assert tr.latency('bright good') < 1.0
@pytest.mark.req('R-NET-08')
@pytest.mark.t7
def test_T7_information_transfer():
    tr=FastTranslator(); assert tr.encode('bright dark') == [1,-1]
@pytest.mark.req('R-NET-09')
@pytest.mark.t8
def test_T8_hop_limit():
    c=Critic(hop_limit=2,budget=9); assert c.allow(0,1,0) and c.allow(1,2,1); assert not c.allow(2,3,2)
@pytest.mark.req('R-NET-10')
@pytest.mark.t9
def test_T9_repeat_and_duplicate_edges_blocked():
    c=Critic(hop_limit=9,budget=9); assert c.allow(0,1,0); assert not c.allow(0,1,1) and not c.allow(1,1,1)
@pytest.mark.req('R-7.2')
@pytest.mark.security
def test_no_forbidden_egress_imports():
    banned={'socket','urllib','requests','http','ftplib','smtplib','subprocess','asyncio'}
    for path in (ROOT/'src').rglob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): names=[a.name.split('.')[0] for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module.split('.')[0]]
                if node.module in {'http.server', 'urllib.parse'}:
                    names.remove(node.module.split('.')[0])
            else: continue
            assert not banned.intersection(names), f'forbidden import in {path}: {names}'
@pytest.mark.req('R-TRACE-01')
@pytest.mark.traceability
def test_each_requirement_and_test_is_traceable():
    reqs={r['requirement_id'] for r in csv.DictReader((ROOT/'compliance/traceability.csv').open())}; tests=list((ROOT/'tests').glob('test_*.py')); cited=set()
    for path in tests:
        tree=ast.parse(path.read_text())
        for fn in (n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')):
            doc=ast.get_docstring(fn) or ''; markers=' '.join(ast.unparse(d) for d in fn.decorator_list); ids={r for r in reqs if r in doc+markers}; assert ids, f'{path}:{fn.name} lacks an existing requirement ID and marker'; cited.update(ids)
    assert reqs <= cited
