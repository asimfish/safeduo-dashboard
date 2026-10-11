import unittest
from fourhand_payload_phases_v1 import Observation,FourHandPayloadPhases
class PreliftTests(unittest.TestCase):
    def test_four_contact_alone_cannot_admit_lift(self):
        controller=FourHandPayloadPhases('beam',(.2,0,1.02),(.2,.04),(.04,.7,.08),.45)
        for i in range(100):
            t=(i+1)*.008333
            obs=Observation('beam',i,t,t,True,(.2,0,1.02),(1,0,0,0),(0,0,0),(0,0,0),(.04,.7,.08),(4,4,4,4),4.4145,(.2,.2,.2,.2),False)
            self.assertEqual(controller.update(obs).phase,'CLOSE')
        for i in range(100,130):
            t=(i+1)*.008333
            obs=Observation('beam',i,t,t,True,(.2,0,1.02),(1,0,0,0),(0,0,0),(0,0,0),(.04,.7,.08),(4,4,4,4),4.4145,(.2,.2,.2,.2),True)
            self.assertEqual(controller.update(obs).phase,'CLOSE')
        for i in range(130,133):
            t=(i+1)*.008333
            obs=Observation('beam',i,t,t,True,(.2,0,1.02),(1,0,0,0),(0,0,0),(0,0,0),(.04,.7,.08),(4,4,4,4),4.4145,(.2,.2,.2,.2),True)
            result=controller.update(obs)
        self.assertEqual(result.phase,'LIFT')
if __name__=='__main__':unittest.main()
