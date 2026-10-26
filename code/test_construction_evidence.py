import unittest
import numpy as np

from construction_evidence_followup import corroborated_prediction, summarize, eligible, choose


class EvidenceTests(unittest.TestCase):
    def test_original_decisions_and_threshold_equality_preserved(self):
        p=np.array([0.,.5,.8,.98,1.]);r=1-p;d=np.zeros(5)
        np.testing.assert_array_equal(corroborated_prediction(p,r,.8,.6,d),p>=.8)
        np.testing.assert_array_equal(corroborated_prediction(p,r,.8,.6,d,'robust'),r>=.6)

    def test_distance_veto_is_monotone_and_has_no_label_input(self):
        p=np.array([.9,.8,.7,.6]);r=np.ones(4);d=np.array([0.,.01,.02,.25])
        previous=corroborated_prediction(p,r,.5,.5,d)
        for minimum in (.01,.02,.04):
            current=corroborated_prediction(p,r,.5,.5,d,minimum_distance_m=minimum)
            self.assertTrue((current<=previous).all());previous=current
        np.testing.assert_array_equal(previous,[0,0,0,1])

    def test_margin_centering_uses_each_models_threshold(self):
        p=np.array([.9,.99,.8]);r=np.array([.3,.01,.99]);d=np.ones(3)*.25
        result=corroborated_prediction(p,r,.9,.3,d,'mean_margin')
        np.testing.assert_array_equal(result,[1,0,1])

    def test_agreement_and_permutation(self):
        p=np.array([.8,.2,.9,.9]);r=np.array([.2,.9,.8,.7]);d=np.array([.1,.1,0.,.1])
        expected=np.array([0,0,0,1]);order=np.array([3,1,0,2])
        np.testing.assert_array_equal(corroborated_prediction(p,r,.5,.5,d,'agreement',.02),expected)
        np.testing.assert_array_equal(corroborated_prediction(p[order],r[order],.5,.5,d[order],'agreement',.02),expected[order])

    def test_unavailable_rows_and_absent_classes_keep_existing_conventions(self):
        data=dict(labels=np.array([1,0,1,0]),available=np.array([1,1,0,1],bool),pair=np.array([1,1,1,1]),scan=np.array([0,0,1,1]))
        scores,rows=summarize(data,np.array([1,0,0,0]))
        self.assertEqual(scores['available_cores'],3)
        self.assertEqual(scores['tp'],1);self.assertEqual(scores['fn'],0)
        self.assertEqual(scores['binary_macro_f1'],.75)

    def test_gates_apply_before_ranking(self):
        reference=dict(clean_binary_macro_f1=.7,clean_changed_f1=.6,clean_false_positive_rate=.1,clean_changed_recall=.8,
                       mean_binary_macro_f1=.65,mean_changed_f1=.55,mean_changed_recall=.7)
        safe=dict(reference,name='safe',order=1,clean_binary_macro_f1=.72,mean_binary_macro_f1=.67)
        unsafe=dict(safe,name='unsafe',order=0,clean_binary_macro_f1=.95,mean_binary_macro_f1=.95,clean_changed_recall=.6)
        self.assertTrue(eligible(safe,reference,'clean'));self.assertFalse(eligible(unsafe,reference,'clean'))
        self.assertEqual(choose([unsafe,safe],reference,'clean')['name'],'safe')
        self.assertIsNone(choose([reference],reference,'robust'))


if __name__=='__main__':unittest.main(verbosity=2)
