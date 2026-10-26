"""Indexed metrics with abstentions retained as errors in full-support scores."""
import numpy as np


def binary_scores(labels,prediction):
    y=np.asarray(labels);p=np.asarray(prediction)
    assert y.shape==p.shape and np.isin(y,[0,1]).all() and np.isin(p,[-1,0,1]).all()
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p!=1)).sum());tn=int(((y==0)&(p==0)).sum())
    unknown_positive=int(((y==1)&(p<0)).sum());unknown_negative=int(((y==0)&(p<0)).sum())
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    f1=2*tp/max(1,2*tp+fp+fn)
    # An abstention is FN for its actual class, not a prediction of unchanged.
    fn_unchanged=fp+unknown_negative
    fp_unchanged=int(((y==1)&(p==0)).sum())
    f1_unchanged=2*tn/max(1,2*tn+fp_unchanged+fn_unchanged)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,unknown_positive=unknown_positive,unknown_negative=unknown_negative,
        precision=precision,recall=recall,f1_changed=f1,iou_changed=tp/max(1,tp+fp+fn),
        f1_unchanged=f1_unchanged,binary_macro_f1=(f1+f1_unchanged)/2,
        accuracy=(tp+tn)/max(1,len(y)),coverage=float(np.mean(p>=0)) if len(p) else 0.,
        changed_prevalence=float(y.mean()) if len(y) else 0.,cores=len(y),positives=int(y.sum()))


def pair_scores(labels,prediction):
    scores=[binary_scores(y,p) for y,p in zip(labels,prediction)]
    result={f'scan{i}_{k}':v for i,s in enumerate(scores) for k,v in s.items()}
    result['directional_changed_f1']=sum(s['f1_changed'] for s in scores)/2
    result['selection_binary_macro_f1']=sum(s['binary_macro_f1'] for s in scores)/2
    result['coverage']=sum(int((p>=0).sum()) for p in prediction)/max(1,sum(len(y) for y in labels))
    result['correct_over_requested']=sum(int((p==y).sum()) for p,y in zip(prediction,labels))/max(1,sum(len(y) for y in labels))
    return result


def balanced_training_weights(y,pair_code):
    """Equal total weight per pair and per class within that pair (training only)."""
    weights=np.zeros(len(y),np.float32)
    groups=np.unique(pair_code)
    for pair in groups:
        mask=pair_code==pair
        present=np.unique(y[mask])
        for label in present:
            selected=mask&(y==label)
            weights[selected]=len(y)/(len(groups)*len(present)*selected.sum())
    return weights


def choose_threshold(labels,probabilities,pair_code,thresholds):
    rows=[]
    for threshold in thresholds:
        macro=[];changed=[]
        for pair in np.unique(pair_code):
            selected=pair_code==pair
            score=binary_scores(labels[selected],(probabilities[selected]>=threshold).astype(np.int8))
            macro.append(score['binary_macro_f1']);changed.append(score['f1_changed'])
        rows.append(dict(threshold=float(threshold),objective=float(np.mean(macro)),changed_f1=float(np.mean(changed))))
    # Fixed tie break: higher precision bias via higher threshold when exactly tied.
    best=max(rows,key=lambda row:(row['objective'],row['threshold']))
    return best,rows


def directional_three_class_scores(labels,predictions):
    """Pooled unchanged/removed/added; unknown is a fourth prediction column."""
    confusion=np.zeros((3,4),np.int64)
    for scan,(y,p) in enumerate(zip(labels,predictions)):
        truth=np.where(y==0,0,scan+1)
        estimate=np.where(p<0,3,np.where(p==0,0,scan+1))
        np.add.at(confusion,(truth,estimate),1)
    result={};f1=[];iou=[]
    for label,name in enumerate(('unchanged','removed','added')):
        tp=int(confusion[label,label]);fp=int(confusion[:,label].sum()-tp);fn=int(confusion[label,:].sum()-tp)
        value=2*tp/max(1,2*tp+fp+fn);intersection=tp/max(1,tp+fp+fn)
        result[f'pooled_{name}_f1']=value;result[f'pooled_{name}_iou']=intersection
        f1.append(value);iou.append(intersection)
    result['pooled_three_class_macro_f1']=float(np.mean(f1));result['pooled_three_class_miou']=float(np.mean(iou))
    return result
