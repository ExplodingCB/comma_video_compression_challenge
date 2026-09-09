"""Repartition the exact model bytes with LZMA2/Brotli/raw, charging block headers."""
from sweep import *
import functools

def main():
    started=time.monotonic()
    cuts={0,len(MODELS),4,263,522,13377,15777,16080,16597,16627,24911,52637,52643,52739,52775,52807,52819}
    cuts.update(range(0,len(MODELS),2048))
    cuts.update(24911+v for v in struct.unpack_from('<15H',MODELS,16597))
    basis_bits=int.from_bytes(MODELS[52637:52640],'little')
    cuts.add(52819+(basis_bits+7)//8)
    for b in BASE: cuts.add(b['start']); cuts.add(b['end'])
    cuts=sorted(c for c in cuts if 0<=c<=len(MODELS))
    configs=[('raw',{})]
    configs += [('brotli',dict(quality=q,mode=m,lgwin=16)) for q,m in ((10,0),(10,2),(11,0))]
    configs += [('lzma',b['filter']) for b in BASE if b['filter']]
    @functools.lru_cache(None)
    def edge(start,end):
        raw=MODELS[start:end]; best=None
        for codec,p in configs:
            print(json.dumps(dict(start=start,end=end,codec=codec,params=p)),flush=True)
            data=encode(raw,codec,p)
            cost=len(data)+4
            if best is None or cost<best[0]: best=(cost,codec,p,data)
        return best
    dist=[float('inf')]*len(cuts); dist[0]=0; prev={}
    for j in range(1,len(cuts)):
        for i in range(j):
            cost,*_=edge(cuts[i],cuts[j])
            if dist[i]+cost<dist[j]: dist[j]=dist[i]+cost; prev[j]=i
        if j%10==0: print(json.dumps(dict(finished=j,total=len(cuts),edges=edge.cache_info().currsize)),flush=True)
    spans=[]; j=len(cuts)-1
    while j:
        i=prev[j]; spans.append((cuts[i],cuts[j])); j=i
    spans.reverse(); blocks=[]
    for start,end in spans:
        cost,codec,p,data=edge(start,end)
        assert decode(data,codec,p,end-start)==MODELS[start:end]
        blocks.append(dict(start=start,end=end,codec=codec,params=p,stored_bytes=len(data)))
    report=dict(archive_bytes=100+9+len(TAIL)+dist[-1],saved_bytes=186151-(100+9+len(TAIL)+dist[-1]),
                blocks=blocks,cuts=len(cuts),edges=edge.cache_info().currsize,elapsed_seconds=time.monotonic()-started)
    (OUT/'dp-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
