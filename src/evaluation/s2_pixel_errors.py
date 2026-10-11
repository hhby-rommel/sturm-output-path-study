# Copyright (C) 2026 STURM output-path study contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Source-release notice added 2026-10-11; numerical code unchanged.
"""Label-grounded pixel errors. Components are annotation fragments, not lakes."""
import numpy as np
from scipy import ndimage


def confusion_counts(target, prediction, valid):
    target=np.asarray(target);prediction=np.asarray(prediction);valid=np.asarray(valid,dtype=bool)
    if target.shape!=prediction.shape or target.shape!=valid.shape:
        raise ValueError('pixel arrays must have identical shapes')
    return np.bincount((2*target[valid]+prediction[valid]).astype(int),minlength=4).reshape(2,2)


def scores(cm):
    cm=np.asarray(cm,dtype=np.int64)
    tn,fp,fn,tp=(int(x) for x in cm.ravel())
    return dict(confusion_matrix=cm.tolist(),valid_pixels=int(cm.sum()),
        tn=tn,fp=fp,fn=fn,tp=tp,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        f1=2*tp/max(2*tp+fp+fn,1),flood_support=tp+fn)


def spatial_errors(target,prediction,valid,reference_valid=None):
    target=np.asarray(target);prediction=np.asarray(prediction);valid=np.asarray(valid,dtype=bool)
    reference=valid if reference_valid is None else np.asarray(reference_valid,dtype=bool)
    if reference.shape!=valid.shape or np.any(valid&~reference):
        raise ValueError('evaluation domain must be inside reference valid domain')
    reference_flood=(target==1)&reference
    flood=reference_flood&valid;pred=(prediction==1)&valid
    structure=np.ones((3,3),dtype=bool)
    interior=ndimage.binary_erosion(reference,structure=structure,iterations=2,border_value=0)
    # Boundary band on either side, 2px from foreground/background interface.
    band=(ndimage.binary_dilation(reference_flood,structure=structure,iterations=2) ^
          ndimage.binary_erosion(reference_flood,structure=structure,iterations=2)) & interior & valid
    fn=flood&~pred;fp=pred&~flood
    labels,n=ndimage.label(reference_flood,structure=structure)
    sizes=np.bincount(labels.ravel(),minlength=n+1)[1:]
    retained=np.bincount(labels[flood].ravel(),minlength=n+1)[1:]
    hits=np.bincount(labels[pred].ravel(),minlength=n+1)[1:]
    bins=[]
    for low,high in ((1,16),(17,64),(65,256),(257,2**31-1)):
        reference_eligible=(sizes>=low)&(sizes<=high)
        eligible=reference_eligible&(retained>0)
        gt=int(retained[eligible].sum());tp=int(hits[eligible].sum())
        bins.append(dict(min_pixels=low,max_pixels=high,components=int(eligible.sum()),
            completely_missed=int((eligible&(hits==0)).sum()),gt_pixels=gt,tp_pixels=tp,
            reference_components=int(reference_eligible.sum()),
            unobservable_components=int((reference_eligible&(retained==0)).sum()),
            reference_gt_pixels=int(sizes[reference_eligible].sum()),
            pixel_recall=tp/gt if gt else None))
    pl,pn=ndimage.label((prediction==1)&reference,structure=structure)
    ps=np.bincount(pl[pred].ravel(),minlength=pn+1)[1:]
    overlaps=np.bincount(pl[flood].ravel(),minlength=pn+1)[1:]
    false=(overlaps==0)&(ps>0)
    return dict(boundary_band_pixels=int(band.sum()),boundary_fp=int((fp&band).sum()),
        boundary_fn=int((fn&band).sum()),interior_nonboundary_fp=int((fp&interior&~band).sum()),
        interior_nonboundary_fn=int((fn&interior&~band).sum()),
        excluded_border_invalid_neighborhood_pixels=int((valid&~interior).sum()),
        label_components=int(n),prediction_components=int(pn),
        no_overlap_prediction_components=int(false.sum()),
        no_overlap_prediction_pixels=int(ps[false].sum()),component_bins=bins)
