"""Prepare an aggressive all12 int4 basis candidate with charged learned data."""
from common import *
def main():
    ctx=context();can=requantize_cpr1_basis(ctx['canonical'],atoms=(0,1,3,4,6,7,8,10,11)).carrier
    data,can=archive(ctx,can,ctx['codes'])
    folder=OUT/'all-int4';folder.mkdir(exist_ok=True)
    (folder/'archive.zip').write_bytes(data);(folder/'canonical.bin').write_bytes(can);np.save(folder/'codes.npy',ctx['codes'])
    print(json.dumps(dict(archive_bytes=len(data),saved_bytes=186151-len(data),archive_sha256=sha(data))),flush=True)
if __name__=='__main__':main()
