import json
import math
from pathlib import Path
import unittest
import torch

from core import generate_graph
from models import InductiveGNN
from intervention import aggregate, forward, perturb


class InterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(9821)
        self.channels={c['name']:c for c in json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())['channels']}
        self.graph=generate_graph(n=32,feature_dimension=8,lambda_=1,rho=.6,mu=.35,seed=101,graph_id='unit',condition='unit')

    def test_native_matches_original_for_both_architectures(self):
        for arch in ('mean','mean_degree'):
            model=InductiveGNN(arch,8,64,1.,.4,1.,.1).eval()
            with torch.no_grad():
                z,_=forward(model,self.graph,self.channels['native'])
                self.assertTrue(torch.equal(z,model(self.graph.x,self.graph.edge_src,self.graph.degree)))

    def test_degree_coordinate_protected_and_empty_means_unchanged(self):
        model=InductiveGNN('mean_degree',8,64,1.,.4,1.,.1).eval()
        g=self.graph
        base,_,_=aggregate(model.layer1,g.x,g.edge_src,g.degree,self.channels['native'],1)
        for channel in self.channels.values():
            out,_,_=aggregate(model.layer1,g.x,g.edge_src,g.degree,channel,1,torch.ones_like(g.x))
            self.assertTrue(torch.equal(out[:,-1],base[:,-1]))
            self.assertTrue(torch.equal(out[g.degree==0,:-1],base[g.degree==0,:-1]))

    def test_scale_frozen_independent_of_realized_degree(self):
        mean=torch.ones(4,8);degree=torch.tensor([0,1,24,48]);noise=torch.ones_like(mean)
        out=perturb(mean,degree,self.channels['gaussian_1'],1,noise)
        expected=torch.ones(8)*24**(-.75);expected[0]*=math.sqrt(1+.35**2)
        self.assertTrue(torch.allclose(out[1]-mean[1],expected))
        self.assertTrue(torch.equal(out[1],out[2]));self.assertTrue(torch.equal(out[2],out[3]))
        self.assertTrue(torch.equal(out[0],mean[0]))

    def test_rounding_happens_only_at_requested_layers(self):
        mean=torch.tensor([[.12345678]*8]);degree=torch.ones(1,dtype=torch.int64)
        for kind in ('float16','bfloat16'):
            out=perturb(mean,degree,self.channels[kind],1)
            self.assertTrue(torch.equal(out,mean.to(getattr(torch,kind)).float()))
            self.assertFalse(torch.equal(out,mean))
            self.assertTrue(torch.equal(perturb(mean,degree,self.channels[kind],2),mean))

    def test_noise_variance_and_common_random_scale_sweep(self):
        mean=torch.zeros(50000,8);degree=torch.ones(50000,dtype=torch.int64)
        z=torch.randn_like(mean)
        large=perturb(mean,degree,self.channels['gaussian_1'],1,z)
        small=perturb(mean,degree,self.channels['gaussian_025'],1,z)
        self.assertTrue(torch.equal(small,large*.25))
        variance=float(large[:,1:].square().mean())
        self.assertLess(abs(variance/(24**(-1.5))-1),.015)
        u=(torch.rand_like(mean)*2-1)*math.sqrt(3)
        uniform=perturb(mean,degree,self.channels['uniform_1'],1,u)
        self.assertLess(abs(float(uniform[:,1:].square().mean())/(24**(-1.5))-1),.015)


if __name__=='__main__': unittest.main()
