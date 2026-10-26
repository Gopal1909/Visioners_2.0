"""Supplementary metrics from audited binary confusion counts; unknowns penalized."""


def supplementary(row):
    def total(key):return sum(int(float(row[f'scan{i}_{key}'])) for i in (0,1))
    tp=total('tp');fp=total('fp');fn=total('fn');tn=total('tn')
    result=dict(pooled_changed_precision=tp/max(1,tp+fp),pooled_changed_recall=tp/max(1,tp+fn),
        pooled_changed_f1=2*tp/max(1,2*tp+fp+fn),pooled_false_positive_rate=fp/max(1,fp+tn+total('unknown_negative')))
    unchanged_fp=fn-total('unknown_positive');unchanged_fn=fp+total('unknown_negative')
    result.update(pooled_unchanged_f1=2*tn/max(1,2*tn+unchanged_fp+unchanged_fn),
        pooled_unchanged_iou=tn/max(1,tn+unchanged_fp+unchanged_fn))
    for scan,name in enumerate(('removed','added')):
        a,b,c=[int(float(row[f'scan{scan}_{k}'])) for k in ('tp','fp','fn')]
        result[f'pooled_{name}_f1']=2*a/max(1,2*a+b+c);result[f'pooled_{name}_iou']=a/max(1,a+b+c)
    result['pooled_three_class_macro_f1']=sum(result[f'pooled_{name}_f1'] for name in ('unchanged','removed','added'))/3
    result['pooled_three_class_miou']=sum(result[f'pooled_{name}_iou'] for name in ('unchanged','removed','added'))/3
    return result
