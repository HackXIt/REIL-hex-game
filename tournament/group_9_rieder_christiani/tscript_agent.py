"""
TorchScript facade agent.
Loads `policy.pt` (exported by export_tscript.py) and chooses a legal move.
"""

from pathlib import Path
import numpy as np, random, torch

# ── lazy-load scripted policy ─────────────────────────────────────────
_policy = None
def _load_policy():
    global _policy
    if _policy is None:
        p = Path(__file__).with_name("policy.pt")
        _policy = torch.jit.load(p, map_location="cpu")
        _policy.eval()
    return _policy

# ── board → tensor (1,3,7,7) ─────────────────────────────────────────
def _obs_tensor(board):
    b = np.asarray(board, np.float32)
    p1  = (b ==  1).astype(np.float32)
    p_1 = (b == -1).astype(np.float32)
    turn= np.full_like(b, 1.0 if (b==1).sum()==(b==-1).sum() else 0.0)
    return torch.tensor(np.stack([p1,p_1,turn], 0))[None]

# ── main callable ────────────────────────────────────────────────────
def tscript_agent(board, action_set):
    """
    Parameters
    ----------
    board : list[list[int]]   - 7 x 7 array with 0/1/-1
    action_set : list[(r,c)]  - legal moves for current player

    Returns one legal (row, col).
    """
    if len(action_set) == 1:            # trivial late-game case
        return action_set[0]

    size   = len(board)
    idx    = lambda rc: rc[0]*size + rc[1]

    obs    = _obs_tensor(board)
    logits = _load_policy()(obs)[0].detach().numpy()

    # mask out illegal indices
    masked = np.full_like(logits, -np.inf, dtype=np.float32)
    for rc in action_set:
        masked[idx(rc)] = logits[idx(rc)]

    if not np.isfinite(masked).any():   # extremely unlikely
        return random.choice(action_set)

    choice = int(np.nanargmax(masked))
    return divmod(choice, size)         # (row,col)
