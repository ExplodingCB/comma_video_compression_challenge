"""Check whether schema splits improve the refined first block further."""
from sweep import *

def main():
    report=json.loads((OUT/'refine-report.json').read_text())
    first=report['blocks'][0]
    configs=[('raw',{})]+list(codec_configs(full=False))
    configs += [('lzma',b['filter']) for b in BASE if b['filter']]
    results=[]
    for cuts in ([0,522,first['end']],[0,13377,first['end']],[0,522,13377,first['end']]):
        blocks=[]
        for a,b in zip(cuts,cuts[1:]):
            raw=MODELS[a:b]; best=None
            for kind in ('identity','transpose:2','transpose:3','transpose:4'):
                altered=transform(raw,kind)
                for codec,p in configs:
                    if codec=='ppmd': continue
                    data=encode(altered,codec,p); cost=len(data)+4+extra(codec,kind)
                    if best is None or cost<best['cost']:
                        assert transform(decode(data,codec,p,len(raw)),kind,True)==raw
                        best=dict(start=a,end=b,codec=codec,kind=kind,params=p,stored_bytes=len(data),cost=cost)
            blocks.append(best)
        results.append(dict(cuts=cuts,blocks=blocks,cost=sum(b['cost'] for b in blocks)))
    (OUT/'split-screen.json').write_text(json.dumps(results,indent=2)+'\n')
    best=min(results,key=lambda r:r['cost'])
    if best['cost']<first['cost']:
        report['blocks']=best['blocks']+report['blocks'][1:]
        report['archive_bytes']-=first['cost']-best['cost']
        report['saved_bytes']=186151-report['archive_bytes']
        report['additional_schema_split_screen']=True
        (OUT/'refine-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
