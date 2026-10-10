"""Behavioral fault and temporal tests; these do not prove native grasp success."""
from dataclasses import replace
import unittest

from payload_phases_v1 import Observation, PayloadPhases


def observation(t, seq, z=.84, x=.5, hands=(1., 1.), table=0., open_error=(.8, .8)):
    return Observation('beam700', seq, t, t, True, (x, 0., z), (1., 0., 0., 0.),
                       (0., 0., 0.), (0., 0., 0.), (.04, .7, .08), hands, table, open_error)


def advance(candidate, until, **kwargs):
    """Supply all observations at10ms intervals, rather than infer missing time."""
    now = 0. if candidate.last_time is None else candidate.last_time + .01
    sequence = candidate.last_sequence + 1
    intent = None
    while now < until - 1e-9:
        intent = candidate.update(observation(now, sequence, **kwargs))
        if intent.phase in ('DONE', 'FAILED'):
            return intent
        sequence += 1
        now += .01
    return candidate.update(observation(until, sequence, **kwargs))


class FeedbackTests(unittest.TestCase):
    def candidate(self):
        return PayloadPhases('beam700', (.5, 0., .84), (.37, 0.), (.04, .7, .08), .45)

    def lifted(self):
        c = self.candidate()
        advance(c, .25)
        advance(c, 2.3, z=.98)
        self.assertEqual(c.phase, 'CARRY')
        return c

    def placed(self):
        c = self.lifted()
        advance(c, 2.9, z=.98, x=.37)
        self.assertEqual(c.phase, 'PLACE')
        advance(c, 3.5, x=.37, table=4.4145)
        self.assertEqual(c.phase, 'RELEASE')
        return c

    def test_height_duration_is_contiguous(self):
        c = self.candidate()
        advance(c, .25)
        advance(c, 1.4, z=.98)
        advance(c, 1.5, z=.93)
        advance(c, 2.8, z=.98)
        self.assertEqual(c.phase, 'LIFT')
        advance(c, 3.6, z=.98)
        self.assertEqual(c.phase, 'CARRY')

    def test_missing_observations_cannot_prove_continuous_hold(self):
        c = self.candidate()
        c.update(observation(0., 0))
        intent = c.update(observation(2., 1, z=.98))
        self.assertEqual(intent.failure, 'observation_gap')
        self.assertTrue(intent.freeze_pair)

    def test_old_50mm_gate_cannot_pass(self):
        c = self.candidate()
        advance(c, 13., z=.90)
        self.assertEqual(c.phase, 'FAILED')
        self.assertEqual(c.failure, 'phase_timeout_LIFT')

    def test_open_command_does_not_prove_release(self):
        c = self.placed()
        intent = advance(c, 5., x=.37, hands=(14., 3.), table=4.4145, open_error=(.88, .83))
        self.assertTrue(intent.open_hands)
        self.assertEqual(intent.phase, 'RELEASE')
        advance(c, 5.6, x=.37, hands=(0., 0.), table=4.4145, open_error=(.01, .01))
        self.assertEqual(c.phase, 'RETREAT')

    def test_overloaded_or_unsupported_placement_does_not_open(self):
        c = self.lifted()
        advance(c, 2.9, z=.98, x=.37)
        intent = advance(c, 4.2, x=.37, table=30.)
        self.assertEqual(intent.phase, 'PLACE')
        self.assertFalse(intent.open_hands)

    def test_airborne_one_hand_loss_and_independent_pair(self):
        failed = self.lifted()
        other = self.lifted()
        intent = advance(failed, 2.6, z=.98, hands=(1., 0.))
        self.assertEqual(intent.failure, 'lost_paired_contact_while_airborne')
        self.assertTrue(intent.freeze_pair)
        self.assertEqual(advance(other, 2.4, z=.98, x=.37).phase, 'CARRY')
        self.assertFalse(advance(other, 2.9, z=.98, x=.37).freeze_pair)

    def test_stale_duplicate_geometry_and_nonfinite_inputs(self):
        bad = [replace(observation(0., 0), now_s=.03),
               replace(observation(0., 0), object_id='beam300'),
               replace(observation(0., 0), position=(float('nan'), 0., .84)),
               replace(observation(0., 0), quaternion_wxyz=(2., 0., 0., 0.)),
               replace(observation(0., 0), size_m=(.04, .3, .08))]
        for obs in bad:
            with self.subTest(obs=obs):
                c = self.candidate()
                self.assertEqual(c.update(obs).phase, 'FAILED')
        c = self.candidate()
        c.update(observation(0., 0))
        self.assertEqual(c.update(observation(0., 0)).failure, 'duplicate_or_nonmonotonic_observation')

    def test_retreat_recontact_fails_without_false_task_success(self):
        c = self.placed()
        advance(c, 4.1, x=.37, hands=(0., 0.), table=4.4145, open_error=(0., 0.))
        self.assertEqual(c.phase, 'RETREAT')
        self.assertEqual(advance(c, 4.2, x=.37, hands=(10., 0.), table=4.4145).failure,
                         'recontact_during_retreat')

    def test_whole_sequence_requires_physical_detachment(self):
        c = self.placed()
        advance(c, 5.2, x=.37, hands=(0., 0.), table=4.4145, open_error=(0., 0.))
        self.assertEqual(c.phase, 'DONE')
        self.assertEqual([x['to_phase'] for x in c.transitions],
                         ['LIFT', 'CARRY', 'PLACE', 'RELEASE', 'RETREAT', 'DONE'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
