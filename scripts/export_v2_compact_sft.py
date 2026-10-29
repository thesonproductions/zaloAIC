#!/usr/bin/env python3
"""Prepare *compact* next-action LoRA data from TRAIN-ONLY trajectories.

Preserves no-lookahead: only prior tool results appear in the prompt. Target
schema is provided as supervised retrieval-oracle during TRAIN; inference
must retrieve its own candidates. No public/private data.
"""
from __future__ import annotations
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lifemate_v2.controller import PLAN_SYSTEM_SFT
from lifemate_v2.intent import norm
from lifemate_v2.retrieval import score_tool
ROOTDATA=ROOT/'runs/v2/train_only'
DEST=ROOTDATA/'mlx_compact'

def records(path):
    for line in path.open(encoding='utf-8'):
        if line.strip():yield json.loads(line)

def compact(item):
    target=item['target_action']
    target_name=target['name']
    transcript=item['gold_transcript']
    all_schemas=item['available_tool_schemas']
    # Use a small selection of distractors plus the correct API. Explicitly mark
    # this as supervised oracle retrieval, NOT a runtime gold/retrieval score.
    ranked=sorted(all_schemas,key=lambda t:(-score_tool(t,transcript),t['name']))
    candidates=ranked[:6]
    if target_name!='__stop__' and target_name not in {x['name'] for x in candidates}:
        selected=next((t for t in all_schemas if t['name']==target_name),None)
        if selected:candidates=candidates[:5]+[selected]
    # For STOP targets, retain the latest actually executed tools so the model
    # can learn not to issue extra writes.
    past=item['prior_observations']
    seen_names={x['tool']['name'] for x in past}
    for t in all_schemas:
        if t['name'] in seen_names and t['name'] not in {c['name'] for c in candidates}:
            candidates=candidates[:5]+[t]
    ctools=[{'name':t['name'],'description':t.get('description','')[:80],
             'parameters':t.get('parameters',{})} for t in candidates]
    hist=[]
    for h in item['history'][-3:]:
        if h.get('role')=='user':
            hist.append({'role':'user','text':h.get('audio_text','')[:180]})
        else:
            results=[]
            for result in h.get('tool_results',[])[:2]:
                results.append(json.dumps(result,ensure_ascii=False,separators=(',',':'))[:400])
            hist.append({'role':'assistant','text':h.get('text','')[:160],'results':results})
    prev=[]
    for h in past[-2:]:
        result=json.dumps(h['result'],ensure_ascii=False,separators=(',',':'))
        # Preserve real IDs and key values by using 1600 chars for each result.
        prev.append({'name':h['tool']['name'],'arguments':h['tool']['arguments'],
                     'result':result[:1000]})
    ctx=item['context']
    payload={'request':transcript,
      'context':{'current_time':ctx.get('current_time'),'timezone':ctx.get('timezone'),
                 'profile':ctx.get('user_profile')},
      'history':hist,'tools':ctools,'observed':prev,
      'task':'Next ONE legal tool call or __stop__; only previous results, no future labels.'}
    response={'tool':target_name,'arguments':target.get('arguments',{})}
    return {'messages':[
        {'role':'system','content':PLAN_SYSTEM_SFT},
        {'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))},
        {'role':'assistant','content':json.dumps(response,ensure_ascii=False,separators=(',',':'))}
    ]}

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for split,out in [('train_dev','train'),('validation','valid')]:
        file=ROOTDATA/f'next_action_{split}.jsonl'
        count=0;chars=[];out_file=DEST/f'{out}.jsonl'
        with out_file.open('w',encoding='utf-8') as writer:
            for row in records(file):
                value=compact(row)
                chars.append(sum(len(m['content']) for m in value['messages']))
                writer.write(json.dumps(value,ensure_ascii=False)+'\n')
                count+=1
        print(out,count,'mean chars',round(sum(chars)/count),'max chars',max(chars),'output',out_file)
if __name__=='__main__':main()
