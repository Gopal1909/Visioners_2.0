import unittest
from learning_v3_calibration_followup import select_point


class OperatingPointTests(unittest.TestCase):
    def test_clean_quality_and_false_alarm_constraints_precede_robustness_rank(self):
        def row(mean,clean,fpr,threshold=.5):
            return dict(mean_macro=mean,mean_changed_f1=.6,clean_macro=clean,clean_fpr=fpr,threshold=threshold,candidate_order=0,checkpoint_order=0)
        rows=[row(.95,.7,.01),row(.94,.85,.08),row(.82,.81,.02),row(.82,.81,.02,.6)]
        chosen=select_point(rows,.80,.05)
        self.assertEqual(chosen['mean_macro'],.82);self.assertEqual(chosen['threshold'],.6)
        self.assertIsNone(select_point(rows,.99,.001))


if __name__=='__main__':unittest.main(verbosity=2)
