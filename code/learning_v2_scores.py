"""Exact indexed scoring kernel without full-cloud or plotting imports."""
import numpy as np
from multipair_metrics import pair_scores

def metric_rows(labels,predictions,common,base,protocol_mask,requested_target,requested_label_counts=None):
    rows=[]
    eligible_labels=[y[m] for y,m in zip(labels,protocol_mask)]
    common_mask=[valid&mask for valid,mask in zip(common,protocol_mask)]
    eligible_count=sum(int(m.sum()) for m in protocol_mask)
    for method,pred in predictions.items():
        unknown=sum(int(((p<0)&mask).sum()) for p,mask in zip(pred,protocol_mask))/max(1,eligible_count)
        for support,masks in [('full_requested',protocol_mask),('common_valid',common_mask)]:
            y=[a[m] for a,m in zip(labels,masks)];p=[a[m] for a,m in zip(pred,masks)]
            row=dict(**base,method=method,support=support,**pair_scores(y,p))
            row['evaluation_coverage']=sum(int(m.sum()) for m in masks)/max(1,eligible_count)
            row['detector_unknown_rate']=unknown
            row['source0_protocol_excluded']=int((~protocol_mask[0]).sum())
            row['target_requested_cores']=requested_target;row['target_retained_cores']=len(labels[1])
            initial_total=len(labels[0])+requested_target
            correct=sum(int(((p==y)&m).sum()) for p,y,m in zip(pred,labels,masks))
            row['correct_over_initial_requested_cores']=correct/max(1,initial_total)
            row['evaluation_coverage_initial_requested_cores']=sum(int(m.sum()) for m in masks)/max(1,initial_total)
            for i in (0,1):
                for label in (0,1):
                    denom=int(((labels[i]==label)&protocol_mask[i]).sum())
                    row[f'scan{i}_gt{label}_evaluation_coverage']=float(((labels[i]==label)&masks[i]).sum()/denom) if denom else None
                    requested=(requested_label_counts[i][label] if requested_label_counts else int((labels[i]==label).sum()))
                    row[f'scan{i}_requested_gt{label}']=requested
                    row[f'scan{i}_retained_gt{label}']=int((labels[i]==label).sum())
                    row[f'scan{i}_protocol_eligible_gt{label}']=denom
                    row[f'scan{i}_evaluated_gt{label}']=int(((labels[i]==label)&masks[i]).sum())
            rows.append(row)
    return rows

