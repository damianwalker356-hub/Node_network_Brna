import numpy as np
import pytest

from brna.proto import NodeNet, make_data, run
from brna.reporting import render


@pytest.mark.req('R-PROT-01')
@pytest.mark.gradient
def test_manual_gradient_numeric_check():
    rng=np.random.default_rng(4); model=NodeNet(rng,d=3,T=1); x,y=make_data(2,rng); params=[model.Win,model.E,model.Ws,model.Wm,model.b,model.wo,model.bo,model.G]
    original=[p.copy() for p in params]
    def loss():
        lo=model.fwd(x)[4]; return float(np.mean(np.logaddexp(0,lo)-y*lo))
    # Check representative output-weight gradient with central differences.
    model.train_step(x,y); analytic=(original[5]-model.wo)/model.opt.lr
    model.wo[...] = original[5]; numerical=np.zeros_like(model.wo); eps=1e-5
    for i in range(model.wo.size):
        model.wo[i]+=eps; plus=loss(); model.wo[i]-=2*eps; minus=loss(); model.wo[i]+=eps; numerical[i]=(plus-minus)/(2*eps)
    rel=np.linalg.norm(analytic-numerical)/max(1e-12,np.linalg.norm(analytic)+np.linalg.norm(numerical)); assert rel < 1e-6
@pytest.mark.req('R-7.1')
@pytest.mark.reporting
def test_run_record_includes_required_metadata(tmp_path,monkeypatch):
    from brna import experiment
    monkeypatch.setattr(experiment,'run',lambda kind,seed,steps,batch,test_samples:{'kind':kind,'seed':seed,'params':1,'test_acc':.5})
    record=experiment.build_record([0,1],steps=1,variants=['vector']); assert record['code_sha256'] and record['git_commit'] is not None and record['seeds']==[0,1] and record['settings']['steps']==1 and record['results'] and record['timestamp']
@pytest.mark.req('R-7.3')
@pytest.mark.reporting
def test_report_requires_limitations_and_all_declared_variants():
    record={'settings':{'variants':['a','b']},'results':[{'kind':'a','seed':0,'params':1,'test_acc':.5}]}
    with pytest.raises(ValueError): render([record])
    assert '## Limitations' in render([{'settings':{'variants':['a']},'results':[{'kind':'a','seed':0,'params':1,'test_acc':.5}]}])
@pytest.mark.req('R-REG-01')
@pytest.mark.regression
@pytest.mark.slow
def test_no_communication_ablation_near_chance():
    scores=[run('nocomm',seed,steps=1200)['test_acc'] for seed in (0,1,2)]; assert all(.45 <= score <= .55 for score in scores)
