"""Continue the two compressed candidates with integer-neighbour recovery."""
from common import *
import sys
import rescue
import build_all_int4

def main():
    meta=ROOT/'results/phase2/renderer'
    cases=[('atom0-linear',[]),('all_metadata_b2',[
        '--source-archive',str(meta/'archives/all_metadata_b2/archive.zip'),
        '--master-frames',str(meta/'masters/all_metadata_b2/masters.uint8'),
        '--seg-errors',str(OUT/'joint-cache/all_metadata_b2-seg.npy'),
        '--warm-start',str(OUT/'rescue-all_metadata_b2/codes.npy')])]
    for name,extra in cases:
        sys.argv=['rescue.py','--candidate',str(OUT/('rescue-'+name)),
            '--output',str(OUT/('refined-'+name)),'--pairs','96','--passes','3','--batch','8',
            '--neighbours','2',*extra]
        rescue.main()
    build_all_int4.main()
    sys.argv=['rescue.py','--candidate',str(OUT/'all-int4'),'--output',str(OUT/'rescue-all-int4'),
              '--pairs','96','--passes','4','--batch','8','--neighbours','2']
    rescue.main()
if __name__=='__main__':main()
