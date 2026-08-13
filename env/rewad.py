from util.motion_util import GRASP_PATTERNS


class Reward:
    def __init__(self, motion_type:str) -> None:
        if motion_type not in GRASP_PATTERNS:
            raise NotImplementedError(f"valid rewards are {GRASP_PATTERNS}")
        self._motion_type = motion_type

        self._reward_scale ={
            "squeeze": 0.1,
        }

    def get_reward_func(self):
        if self._motion_type == "squeeze":
            return self._reward_squeeze


    def _reward_squeeze(self,contact_func):
        _,contacts = contact_func()

        hits = contacts.sum()*self._reward_scale["squeeze"]
        return hits
    


        