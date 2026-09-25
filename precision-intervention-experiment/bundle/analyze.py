from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def csv_write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def bootstrap(values, count, seed, nested=True):
    blocks,graphs=values.shape
    rng=np.random.default_rng(seed)
    chosen_blocks=rng.integers(blocks,size=(count,blocks))
    if not nested: return values.mean(axis=1)[chosen_blocks].mean(axis=1)
    chosen_graphs=rng.integers(graphs,size=(count,blocks,graphs))
    return values[chosen_blocks[:,:,None],chosen_graphs].mean(axis=(1,2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();root=args.results
    config=json.loads((root/'config.resolved.json').read_text())
    execution=json.loads((root/'execution.json').read_text());assert execution['status']=='valid'
    with (root/'replicate_metrics.csv').open() as f: raw=list(csv.DictReader(f))
    metrics=['log_loss','accuracy','brier','class_disagreement','layer1_noise_rms','layer1_mean_rms','layer2_noise_rms','layer2_mean_rms']
    groups=defaultdict(list)
    for row in raw: groups[(int(row['block_seed']),row['condition'],int(row['graph_index']),row['architecture'],row['channel'])].append(row)
    graph_rows=[];lookup={}
    for key,rows in sorted(groups.items()):
        block,condition,index,arch,channel=key
        record=dict(block_seed=block,condition=condition,graph_index=index,architecture=arch,channel=channel,noise_replicates=len(rows))
        for metric in metrics:record[metric]=float(np.mean([float(r[metric]) for r in rows]))
        graph_rows.append(record);lookup[key]=record
    csv_write(root/'graph_metrics.csv',graph_rows)
    seeds=sorted(set(r['block_seed'] for r in graph_rows))
    indices=sorted(set(r['graph_index'] for r in graph_rows))
    expected=len(seeds)*len(indices)*len(config['conditions'])*len(config['architectures'])*len(config['channels'])
    assert len(graph_rows)==expected
    if not execution['smoke']:
        assert set(seeds)==set(config['block_seeds']);assert len(indices)==config['graphs_per_condition']
        for row in graph_rows:
            kind=next(c['kind'] for c in config['channels'] if c['name']==row['channel'])
            assert row['noise_replicates']==(config['random_replicates'] if kind in ('gaussian','uniform') else 1)
    def array(cond,arch,channel,metric):
        return np.array([[lookup[(seed,cond,index,arch,channel)][metric] for index in indices] for seed in seeds])
    summaries=[];contrasts=[];block_rows=[]
    def summarize(values,primary):
        samples=bootstrap(values,config['bootstrap_replicates'],config['bootstrap_seed'])
        block_samples=bootstrap(values,config['bootstrap_replicates'],config['bootstrap_seed'],False)
        q=config['primary_quantiles'] if primary else [.025,.975]
        lo,hi=np.quantile(samples,q);blo,bhi=np.quantile(block_samples,q)
        return dict(estimate=float(values.mean()),lower=float(lo),upper=float(hi),
                    block_only_lower=float(blo),block_only_upper=float(bhi),interval_percent=100*(q[1]-q[0]))
    for cond in config['conditions']:
        for ch in config['channels']:
            channel=ch['name']
            for arch in config['architectures']:
                row=dict(condition=cond,channel=channel,architecture=arch)
                for metric in metrics:row[metric]=float(array(cond,arch,channel,metric).mean())
                summaries.append(row)
            for metric in ('log_loss','accuracy','brier'):
                mean=array(cond,'mean',channel,metric);degree=array(cond,'mean_degree',channel,metric)
                base_mean=array(cond,'mean','native',metric);base_degree=array(cond,'mean_degree','native',metric)
                for name,values in [('mean_change',mean-base_mean),('degree_change',degree-base_degree),
                                    ('degree_advantage',mean-degree),('degree_protection',(mean-base_mean)-(degree-base_degree))]:
                    primary=(cond in ('lambda_low','lambda_high') and channel==config['primary_channel']
                             and metric=='log_loss' and name in ('mean_change','degree_protection'))
                    row=dict(condition=cond,channel=channel,metric=metric,contrast=name,primary=primary,**summarize(values,primary))
                    contrasts.append(row)
                    for seed,block_value in zip(seeds,values.mean(axis=1)):
                        block_rows.append(dict(block_seed=seed,condition=cond,channel=channel,metric=metric,contrast=name,estimate=float(block_value)))
    csv_write(root/'summary.csv',summaries);csv_write(root/'contrasts.csv',contrasts);csv_write(root/'block_contrasts.csv',block_rows)
    primary=[r for r in contrasts if r['primary']];assert len(primary)==4
    margin=config['margin_nats'];protections=[r for r in primary if r['contrast']=='degree_protection']
    if execution['smoke']:verdict='smoke_only'
    elif all(r['lower']>margin for r in primary):verdict='supported'
    elif all(r['lower']>=-margin and r['upper']<=margin for r in protections):verdict='practically_negligible_protection'
    elif any(r['upper'] < -margin for r in protections):verdict='evidence_against_protection_at_a_shift'
    else:verdict='mixed_or_inconclusive'
    decision={'execution_valid':True,'verdict':verdict,'margin_nats':margin,'primary_tests':primary,
              'interpretation':'Differential intervention sensitivity conditional on frozen trained models; does not identify rational-denominator encoding.',
              'frozen_sha256':execution['frozen_sha256'],'graph_rows':len(graph_rows),'replicate_rows':len(raw)}
    (root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.name!='RESULT_MANIFEST.json'}
    (root/'RESULT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
