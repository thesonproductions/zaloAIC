#!/usr/bin/env python3
"""Audit contest cutoff and parameter limits of staged HF safetensors without loading model."""
from __future__ import annotations
import datetime as dt,json,struct
from math import prod
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/'model_bundle'
LIMIT_LLM=5_000_000_000
LIMIT_TOTAL=6_000_000_000
CUTOFF=dt.datetime.fromisoformat('2026-10-26T23:59:59+00:00')

def count_params(model_dir):
    n=0
    for file in sorted(model_dir.glob('*.safetensors')):
        with file.open('rb') as f:
            header_bytes=f.read(8)
            if len(header_bytes)!=8:raise RuntimeError('Invalid safetensors: '+str(file))
            length=struct.unpack('<Q',header_bytes)[0]
            if length>50_000_000:raise ValueError('Suspicious header')
            header=json.loads(f.read(length))
            for key,meta in header.items():
                if key=='__metadata__':continue
                n+=prod(meta['shape'])
    return n
def main():
    p=json.loads((MODELS/'provenance.json').read_text())
    for type in ('llm','asr'):
        if dt.datetime.fromisoformat(p[type]['date'])>CUTOFF:
            raise AssertionError('Checkpoint after contest cutoff: '+type)
    a=count_params(MODELS/'llm')
    b=count_params(MODELS/'asr')
    if a<=0 or b<=0:raise AssertionError('Missing HF safetensors')
    if a>=LIMIT_LLM or a+b>=LIMIT_TOTAL:raise AssertionError(f'Param limit exceeded: {a}+{b}')
    print(json.dumps({'llm_parameters':a,'asr_parameters':b,'total_parameters':a+b,
                      'llm_lt_5b':a<LIMIT_LLM,'total_lt_6b':a+b<LIMIT_TOTAL,
                      'model_cutoff_pass':True},indent=2))
if __name__=='__main__':main()
