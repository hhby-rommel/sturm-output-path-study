"""Fixed reference boundary/off-boundary/excluded FP-FN partitions; no inference."""
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

def reference_regions(target, original_valid, evaluation_valid, radius):
    target=np.asarray(target);original_valid=np.asarray(original_valid,dtype=bool)
    evaluation_valid=np.asarray(evaluation_valid,dtype=bool)
    assert radius in (1,2,3)
    assert target.ndim==2 and target.shape==original_valid.shape==evaluation_valid.shape
    original=original_valid & (target!=255)
    assert not np.any(evaluation_valid & ~original)
    water=original & (target==1)
    contour=water & ~binary_erosion(water,structure=np.ones((3,3),bool),border_value=0)
    coords=np.arange(-radius,radius+1)
    disk=coords[:,None]**2+coords[None,:]**2<=radius**2
    band=binary_dilation(contour,structure=disk)
    # Reference geometry is fixed before perturbation deletion. Margin includes
    # the contour's 8-neighbour support plus the distance tolerance.
    eligible=binary_erosion(original,structure=np.ones((2*(radius+1)+1,)*2,bool),border_value=0)
    domain=evaluation_valid & original
    regions=dict(boundary=domain & eligible & band,
                 off_boundary=domain & eligible & ~band,
                 excluded_neighbourhood=domain & ~eligible)
    assert np.array_equal(sum(q.astype(np.uint8) for q in regions.values()),domain.astype(np.uint8))
    return regions

def partition_counts(target,prediction,original_valid,evaluation_valid,radius):
    target=np.asarray(target);prediction=np.asarray(prediction)
    assert prediction.shape==target.shape and np.isin(prediction,[0,1]).all()
    regions=reference_regions(target,original_valid,evaluation_valid,radius)
    domain=np.asarray(evaluation_valid,dtype=bool) & np.asarray(original_valid,dtype=bool) & (target!=255)
    errors=dict(TP=(target==1)&(prediction==1),FP=(target==0)&(prediction==1),FN=(target==1)&(prediction==0))
    totals={k:int((v&domain).sum()) for k,v in errors.items()}
    rows=[]
    for name,region in regions.items():
        counts={k:int((v&region).sum()) for k,v in errors.items()}
        rows.append(dict(region=name,radius=radius,valid_pixels=int(region.sum()),
                         reference_foreground=int(((target==1)&region).sum()),**counts,
                         signed_imbalance=counts['FP']-counts['FN'],
                         FP_share=counts['FP']/totals['FP'] if totals['FP'] else None,
                         FN_share=counts['FN']/totals['FN'] if totals['FN'] else None))
    assert all(sum(q[k] for q in rows)==totals[k] for k in totals)
    assert sum(q['signed_imbalance'] for q in rows)==totals['FP']-totals['FN']
    return rows,totals

def self_check():
    gt=np.zeros((31,31),np.uint8);gt[8:23,8:23]=1
    valid=np.ones_like(gt,bool);pred=gt.copy()
    pred[7,15]=1;pred[2,15]=1;pred[0,15]=1
    pred[8,15]=0;pred[15,15]=0
    # Known errors occupy boundary, off-boundary and excluded edge neighbourhoods.
    primary,totals=partition_counts(gt,pred,valid,valid,2)
    assert totals==dict(TP=223,FP=3,FN=2)
    assert [(q['FP'],q['FN']) for q in primary]==[(1,1),(0,1),(2,0)]
    deleted=valid.copy();deleted[15,15]=False
    for radius in (1,2,3):
        baseline=reference_regions(gt,valid,valid,radius)
        changed=reference_regions(gt,valid,deleted,radius)
        assert all(np.array_equal(changed[k],v&deleted) for k,v in baseline.items())
        _,totals=partition_counts(gt,pred,valid,deleted,radius)
        assert totals==dict(TP=223,FP=3,FN=1)
    invalid=valid.copy();invalid[15,15]=False
    regions=reference_regions(gt,invalid,invalid,2)
    assert regions['excluded_neighbourhood'][14,15]
    return dict(status='PASS',cases=['known exhaustive error counts','deletion does not redefine reference contour','invalid-neighbourhood exclusion'],
                radii=[2,1,3],model_inference=False,real_prediction_access=False)

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true',required=True);p.parse_args()
    print(json.dumps(self_check()))
