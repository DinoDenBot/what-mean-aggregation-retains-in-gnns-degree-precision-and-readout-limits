from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from core import generate_graph, generate_pool, graph_seed, stable_seed, tensor_state_sha256
from models import InductiveGNN
from intervention import forward

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readcsv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def writejson(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def check_frozen():
    manifest = json.loads((ROOT / 'FROZEN.json').read_text())
    for name, digest in manifest['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Frozen file changed: {name}')
    return sha(ROOT / 'FROZEN.json')


def load_model(seed, arch, original, inventory):
    checkpoint = torch.load(ROOT / 'inputs/checkpoints' / f'block-{seed}-{arch}.pt', map_location='cpu', weights_only=True)
    model = InductiveGNN(arch, 8, checkpoint['architecture']['hidden_width'], 0., 1., 1., original['model']['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = tensor_state_sha256(model.state_dict())
    assert digest == inventory[(seed, arch)]['state_sha256']
    return model, digest


def metrics(logits, labels, baseline):
    z = logits.double(); y = labels.double(); prob = z.sigmoid()
    return {'log_loss': float(F.binary_cross_entropy_with_logits(z, y)),
            'accuracy': float(((z >= 0) == (y >= .5)).double().mean()),
            'brier': float((prob-y).square().mean()),
            'class_disagreement': float(((z >= 0) != (baseline >= 0)).double().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config.json').read_text())
    original = json.loads((ROOT / 'inputs/config.resolved.json').read_text())
    frozen_sha = 'not_frozen_smoke' if args.smoke else check_frozen()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Protocol runtime ceiling')))
    signal.alarm(config['max_runtime_seconds'])
    torch.set_num_threads(config['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    inventory = {(int(r['block_seed']), r['architecture']): r for r in readcsv(ROOT/'inputs/model_inventory.csv')}
    old_metrics = {(int(r['block_seed']), r['condition'], int(r['graph_index']), r['architecture']): r
                   for r in readcsv(ROOT/'inputs/graph_metrics.csv')}
    conditions = {r['condition']:r for r in original['conditions']}
    seeds = config['block_seeds'][:1] if args.smoke else config['block_seeds']
    graph_count = 1 if args.smoke else config['graphs_per_condition']
    repetitions = 1 if args.smoke else config['random_replicates']
    native = config['channels'][0]
    graph_inventory = []; weight_hashes = {}; maximum_native_error = 0.; maximum_old_metric_error = 0.
    rows_written = 0; graph_total = len(seeds)*len(config['conditions'])*graph_count
    with (args.output/'replicate_metrics.csv').open('w', newline='') as stream:
        fields = ['block_seed','condition','graph_index','graph_seed','architecture','channel','replicate',
                  'log_loss','accuracy','brier','class_disagreement','layer1_noise_rms','layer1_mean_rms',
                  'layer2_noise_rms','layer2_mean_rms']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for seed in seeds:
            models = {}
            for arch in config['architectures']:
                models[arch], weight_hashes[f'{seed}:{arch}'] = load_model(seed, arch, original, inventory)
            for condition in config['conditions']:
                # Reproduce a pre-existing target before using new graphs; not scientific evidence for the intervention.
                old = generate_pool(original, block_seed=seed, pool='target', condition=conditions[condition], count=1)[0]
                for arch, model in models.items():
                    with torch.inference_mode():
                        old_z = model(old.x, old.edge_src, old.degree)
                    old_values = {'log_loss': float(F.binary_cross_entropy_with_logits(old_z, old.y)),
                                  'accuracy':float(((old_z.sigmoid()>=.5)==(old.y>=.5)).float().mean()),
                                  'brier':float((old_z.sigmoid()-old.y).square().mean())}
                    for metric, value in old_values.items():
                        error = abs(value-float(old_metrics[(seed,condition,0,arch)][metric]))
                        maximum_old_metric_error = max(maximum_old_metric_error, error)
                        assert error < 2e-6, ('old metric',seed,condition,arch,metric,error)
                del old
                for index in range(graph_count):
                    pool = 'non_result_smoke' if args.smoke else config['target_pool']
                    gseed = graph_seed(config['experiment_id'],seed,pool,condition,index)
                    assert all(gseed != graph_seed(original['experiment_id'],s,p,c['condition'],i)
                               for s in original['training_blocks']['seeds'] for p in ('target','source_train','source_validation')
                               for c in original['conditions'] for i in range(8))
                    c = conditions[condition]
                    graph = generate_graph(n=512 if args.smoke else 4096,feature_dimension=8,
                        lambda_=c['lambda'],rho=c['rho'],mu=c['mu'],seed=gseed,
                        graph_id=f'{pool}:{seed}:{condition}:{index}:{gseed}',condition=condition)
                    graph_inventory.append({'block_seed':seed,'condition':condition,'graph_index':index,'seed':gseed,
                        'n':graph.n,'mean_degree':float(graph.degree.double().mean()),'graph_id':graph.graph_id})
                    baselines = {}
                    for arch, model in models.items():
                        with torch.inference_mode():
                            baselines[arch] = model(graph.x,graph.edge_src,graph.degree)
                            z,_ = forward(model, graph, native)
                        error = float((z-baselines[arch]).abs().max()); maximum_native_error=max(maximum_native_error,error)
                        assert error == 0., ('native mismatch',error)
                    for channel in config['channels']:
                        random = channel['kind'] in ('gaussian','uniform')
                        for replicate in range(repetitions if random else 1):
                            noise = None
                            if random:
                                generator=torch.Generator().manual_seed(stable_seed(config['experiment_id'],gseed,replicate,channel['kind']))
                                noise=(torch.randn(graph.x.shape,generator=generator) if channel['kind']=='gaussian'
                                       else (torch.rand(graph.x.shape,generator=generator)*2-1)*math.sqrt(3))
                            for arch,model in models.items():
                                with torch.inference_mode():
                                    z,diagnostics=forward(model,graph,channel,noise)
                                values=metrics(z,graph.y,baselines[arch])
                                assert all(math.isfinite(v) for v in (*values.values(),*diagnostics.values()))
                                writer.writerow(dict(block_seed=seed,condition=condition,graph_index=index,graph_seed=gseed,
                                    architecture=arch,channel=channel['name'],replicate=replicate,**values,**diagnostics))
                                rows_written+=1
                    stream.flush()
                    event={'graphs_completed':len(graph_inventory),'graphs_total':graph_total,'rows':rows_written,
                           'elapsed_seconds':round(time.monotonic()-started,2),'block_seed':seed,'condition':condition}
                    print(json.dumps(event),flush=True)
                    with (args.output/'progress.jsonl').open('a') as progress: progress.write(json.dumps(event)+'\n')
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8*1024**3:
                        raise RuntimeError('Memory ceiling exceeded')
            for arch,model in models.items():
                assert tensor_state_sha256(model.state_dict()) == weight_hashes[f'{seed}:{arch}']
    expected_per_arch=sum(repetitions if c['kind'] in ('gaussian','uniform') else 1 for c in config['channels'])
    assert rows_written==graph_total*len(config['architectures'])*expected_per_arch
    if not args.smoke: assert check_frozen()==frozen_sha
    writejson(args.output/'graphs.json',graph_inventory)
    writejson(args.output/'weights_verified.json',weight_hashes)
    writejson(args.output/'config.resolved.json',config)
    writejson(args.output/'execution.json',{'status':'valid','smoke':args.smoke,'frozen_sha256':frozen_sha,
        'max_native_logit_error':maximum_native_error,'max_old_metric_error':maximum_old_metric_error,
        'graphs':graph_total,'replicate_rows':rows_written,'weight_hashes_verified':len(weight_hashes),
        'elapsed_seconds':time.monotonic()-started,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'threads':torch.get_num_threads(),'platform':platform.platform(),'pid':os.getpid()})
    print('EXECUTION_VALID',flush=True)


if __name__ == '__main__': main()
