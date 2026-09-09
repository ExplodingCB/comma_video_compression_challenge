"""Exercise changing model lengths without changing recovered bytes."""
import json
import copy
from compose import ROOT,LOSSLESS,adapt_recipe,block_dispatch,build_payload,decode_model_blocks,hybrid_fingerprints,inside

def main():
    original=(ROOT/'results/lossless/models.bin').read_bytes()
    tail=(ROOT/'results/lossless/tail.bin').read_bytes()
    recipe=json.loads((LOSSLESS/'recipe.json').read_text())
    for size in (100,16264,65021,65022,70000,74860,74861):
        models=(original+b'\0')[:size]
        adjusted=adapt_recipe(models,recipe)
        assert adjusted['blocks'][-1]['end']==size
        assert decode_model_blocks(build_payload(models,tail,adjusted))==(models,tail)
    assert recipe==json.loads((LOSSLESS/'recipe.json').read_text())
    try:inside(ROOT.parent/'outside_composition')
    except ValueError:pass
    else:raise AssertionError('Outside-project path accepted')
    for quote in ('"',"'"):
        # Custom-reader logic before and after the container dispatch is kept.
        old=f'ETP1_custom_before\nouter.startswith(b{quote}BLK1{quote})\nETP1_custom_after'
        expected="ETP1_custom_before\nouter.startswith((b'BLK1', b'BLK2'))\nETP1_custom_after"
        assert block_dispatch(old)==expected
        assert block_dispatch(expected)==expected
    try:block_dispatch('unknown container reader')
    except ValueError:pass
    else:raise AssertionError('Unknown reader dispatch accepted')
    models=dict(hpac_blob='same_digest',hpac_bytes_hex='112233',semantic_blob='model_renderer',
                carrier_blob='model_pose',renderer_tensors={'weight':'model_weight'},table_shape=[25,5])
    entropy=dict(hpac_blob='same_digest',hpac_bytes_hex='112233',semantic_blob='entropy_renderer',
                 carrier_blob='entropy_pose',renderer_tensors={'weight':'entropy_weight'},table_shape=[50,5],
                 table_margins=[4],token_stream='entropy_tokens',residual_payload='entropy_table')
    original_models=copy.deepcopy(models);original_entropy=copy.deepcopy(entropy)
    hybrid=hybrid_fingerprints(models,entropy)
    assert hybrid['semantic_blob']=='model_renderer' and hybrid['carrier_blob']=='model_pose'
    assert hybrid['renderer_tensors']==models['renderer_tensors']
    assert hybrid['table_shape']==[50,5] and hybrid['table_margins']==[4]
    assert hybrid['token_stream']=='entropy_tokens' and hybrid['residual_payload']=='entropy_table'
    assert 'hpac_bytes_hex' not in hybrid
    assert models==original_models and entropy==original_entropy
    for key in ('hpac_bytes_hex','hpac_blob'):
        wrong=copy.deepcopy(entropy);wrong[key]='different'
        try:hybrid_fingerprints(models,wrong)
        except ValueError:pass
        else:raise AssertionError(f'Hybrid accepted incompatible {key}')
    print('Passed model-length round trips, immutable recipes, containment, custom-reader preservation, and HPAC-compatible hybrid state checks.')

if __name__=='__main__':main()
