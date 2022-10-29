import numpy as np
import torch 
import multiprocess as mp
from collections.abc import Iterable 
from robot.objective import Objective
from robot.tasks.lunar_lander.lunar_lander_utils import simulate_lunar_lander


class LunarLanderObjective(Objective):
    ''' Lunar Lander optimization task
        Goal is to find a policy for the Lunar Lander 
        smoothly lands on the moon without crashing, 
        thereby maximizing reward 
    ''' 
    def __init__(
        self,
        xs_to_scores_dict={},
        num_calls=0,
        seed=np.arange(50),
        tau=None,
        **kwargs,
    ):
        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict,
            num_calls=num_calls,
            task_id='lunar',
            dim=12,
            lb=0.0,
            ub=1.0,
            **kwargs
        ) 
        self.pool = mp.Pool(mp.cpu_count())
        seed = [seed] if not isinstance(seed, Iterable) else seed 
        self.seed = seed 
        self.dist_func = torch.nn.PairwiseDistance(p=2)


    def query_oracle(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy() 
        x = x.reshape((-1, self.dim))  # bsz x 12 (1, 12)
        ns = len(self.seed) # default 50 
        nx = x.shape[0] # bsz = 1 if pass in one policy x at a time 
        x_tiled = np.tile(x, (ns, 1)) # ns x dim  (10 seds x 12 dim )
        seed_rep = np.repeat(self.seed, nx) # repeat ns x number of policies (bsz) = (ns,) when bsz is 1
        params = [[xi, si] for xi, si in zip(x_tiled, seed_rep)]
        # list with pairs of x's and seeds 
        # so for a single s, we have a list with [(x, s1), (x,s2), ... (x,sN)] 
        # sumulates lunar lander w/ each pair of (x, seed)
        rewards = np.array(self.pool.map(simulate_lunar_lander, params)).reshape(-1)
        # Compute the average score across the seeds 
        mean_reward = np.mean(rewards, axis=0).squeeze()

        return mean_reward


    def divf(self, x1, x2 ):
        return self.dist_func(x1.cuda(), x2.cuda())

