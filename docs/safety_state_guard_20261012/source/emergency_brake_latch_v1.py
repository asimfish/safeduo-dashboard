"""Fail-closed mode transition; plant stopping must be measured independently."""
class EmergencyBrakeLatch:
    def __init__(self):
        self.trigger_control=None

    def update(self,control,decision_mode):
        if type(control) is not int or control<0:raise ValueError('invalid control')
        if decision_mode=='UNQUALIFIED_BRAKE' and self.trigger_control is None:
            self.trigger_control=control
        return self.trigger_control is not None
