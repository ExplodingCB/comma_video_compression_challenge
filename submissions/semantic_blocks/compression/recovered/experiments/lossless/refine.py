"""Coordinate-refine boundaries and layouts around measured winning blocks."""
from sweep import *
import functools

def main():
    started=time.monotonic(); count=0
    configs=[('raw',{})]
    configs += [('brotli',dict(quality=q,mode=m,lgwin=18)) for q,m in ((10,0),(10,2),(11,0))]
    configs += [('lzma',b['filter']) for b in BASE if b['filter']]
    @functools.lru_cache(None)
    def edge(start,end):
        nonlocal count
        raw=MODELS[start:end]; best=None
        for kind in ('identity','transpose:2','transpose:3','transpose:4'):
            changed=transform(raw,kind)
            for codec,p in configs:
                if codec=='raw' and kind!='identity': continue
                data=encode(changed,codec,p); count+=1
                cost=len(data)+4+extra(codec,kind)
                if best is None or cost<best['cost']:
                    assert transform(decode(data,codec,p,len(raw)),kind,True)==raw
                    best=dict(start=start,end=end,codec=codec,kind=kind,params=p,stored_bytes=len(data),cost=cost)
        return best
    cuts=[0,16038,24895,52637,65022,len(MODELS)]
    for radius,step in ((256,16),(32,1)):
        for i in range(1,len(cuts)-1):
            old=cuts[i]; bestcost=edge(cuts[i-1],old)['cost']+edge(old,cuts[i+1])['cost']; chosen=old
            for split in range(max(cuts[i-1]+1,old-radius),min(cuts[i+1],old+radius+1),step):
                cost=edge(cuts[i-1],split)['cost']+edge(split,cuts[i+1])['cost']
                if cost<bestcost: chosen=split; bestcost=cost
            cuts[i]=chosen
            blocks=[edge(a,b) for a,b in zip(cuts,cuts[1:])]
            report=dict(archive_bytes=100+9+len(TAIL)+sum(b['cost'] for b in blocks),
                        blocks=blocks,candidates=count,edges=edge.cache_info().currsize,
                        elapsed_seconds=time.monotonic()-started)
            report['saved_bytes']=186151-report['archive_bytes']
            (OUT/'refine-report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
