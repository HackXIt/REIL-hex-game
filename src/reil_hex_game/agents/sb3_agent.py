# facade_sb3.py
from pathlib import Path
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

_model = None
def _load():
    global _model
    if _model is None:
        zip_path = Path(__file__).with_name("model.zip")   # copy of your checkpoint
        _model = MaskablePPO.load(zip_path, device="cpu")
        _model.set_training_mode(False)                   # disable dropout etc.
    return _model

def _obs_tensor(board):
    b = np.asarray(board, dtype=np.float32)
    p1 = (b == 1).astype(np.float32)
    p2 = (b == -1).astype(np.float32)
    turn = np.full_like(b, 1.0 if (b==1).sum()==(b==-1).sum() else 0.0)
    return np.stack([p1, p2, turn], 0)

def sb3_agent(board, action_set):
    model = _load()
    obs   = _obs_tensor(board)[None]              # batch=1
    masks = np.zeros(49, bool); size=7
    for r,c in action_set: masks[r*size+c] = True
    action, _ = model.predict(obs, action_masks=masks[None], deterministic=True)
    r, c = divmod(int(action[0]), size)
    return (r, c)
