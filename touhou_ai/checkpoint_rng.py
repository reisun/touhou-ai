"""Loading weights must not restart the running collector's random streams."""
import random
import numpy as np
import torch

def capture_rng():
    n=np.random.get_state()
    p=random.getstate()
    return dict(python=[p[0],list(p[1]),p[2]],numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],torch=torch.get_rng_state().tolist())

def restore_rng(state):
    def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,(list,tuple)) else x
    random.setstate(tuples(state['python']))
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),int(n[2]),int(n[3]),float(n[4])))
    torch.set_rng_state(torch.tensor(state['torch'],dtype=torch.uint8))

def load_preserving_rng(algorithm,*args,**kwargs):
    python_state=random.getstate()
    numpy_state=np.random.get_state()
    torch_state=torch.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return algorithm.load(*args,**kwargs)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)
