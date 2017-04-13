from filters.cons import cons
from TestBase import TestBase
import numpy as np


class TestConsFilter(TestBase):
    def testFacts(self):
        np.testing.assert_allclose(cons(np.array([])), np.array([1]))
        np.testing.assert_allclose(cons(np.array([1])), np.array([1, -2.718281828459046]))
        np.testing.assert_allclose(cons(np.array([2])), np.array([1, -7.389056098930650]))
        np.testing.assert_allclose(cons(np.array([3])), np.array([1, -20.085536923187668]))

    def testInputsFromHrfParameters_1(self):
        input = np.array([-0.27, -0.27, -0.4347 - 0.3497j, -0.4347 + 0.3497j])
        expected_output = [1, -2.743302542144524, 2.859320053352163, -1.348960571338628, 0.244289813077911]
        np.testing.assert_allclose(cons(input), expected_output)

    def testInputsFromHrfParameters_2(self):
        input = np.array([-0.1336])
        expected_output = [1, -0.874939970605710]
        np.testing.assert_allclose(cons(input), expected_output)

    def testInputsFromHrfParameters_3(self):
        input = np.array([-0.27, -0.27, -0.4347 - 0.3497j, -0.4347 + 0.3497j, 0])
        expected_output = [1, -3.743302542144524, 5.602622595496687, -4.208280624690792, 1.593250384416538,
                           -0.244289813077911]
        np.testing.assert_allclose(cons(input), expected_output)

    def testInputsFromHrfParameters_4(self):
        input = np.array([-1.020408163265306, -3.092145949288806, -0.324675324675325 - 0.548716683350910j,
                          -0.324675324675325 + 0.548716683350910j])
        expected_output = [1, -1.639165427154841, 1.039293687162910, -0.232195359716452, 0.008549309479686]
        np.testing.assert_allclose(cons(input), expected_output)

    def testInputsFromHrfParameters_5(self):
        input = np.array([-11.898107445013801])
        expected_output = [1, -6.803268190346306e-06]
        np.testing.assert_allclose(cons(input), expected_output)

    def testInputsFromHrfParameters_6(self):
        input = np.array([-1.020408163265306, -3.092145949288806, -0.324675324675325 - 0.548716683350910j,
                          -0.324675324675325 + 0.548716683350910j, 0])
        expected_output = [1, -2.639165427154841, 2.678459114317751, -1.271489046879362, 0.240744669196138,
                           -0.008549309479686]
        np.testing.assert_allclose(cons(input), expected_output)
