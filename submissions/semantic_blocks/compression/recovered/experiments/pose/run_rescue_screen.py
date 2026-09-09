"""Bounded3pass96pair recovery of two extra atoms and an unchanged-basis control."""
from common import *
import sys
import rescue

def main():
    ctx=context();control=OUT/'control';control.mkdir(exist_ok=True)
    (control/'archive.zip').write_bytes(ctx['data']);(control/'canonical.bin').write_bytes(ctx['canonical'])
    np.save(control/'codes.npy',ctx['codes'])
    for name in ['control','atom0-linear','atom7-unchanged']:
        sys.argv=['rescue.py','--candidate',str(OUT/name),'--output',str(OUT/('rescue-'+name)),
                  '--pairs','96','--passes','3','--batch','8']
        rescue.main()
if __name__=='__main__':main()
