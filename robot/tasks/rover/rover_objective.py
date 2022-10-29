import torch 
from robot.objective import Objective
from gpytorch.kernels.kernel import Distance
from robot.tasks.rover.rover_utils import create_large_domain, ConstantOffsetFn


class RoverObjective(Objective):
    ''' Rover optimization task
        Goal is to find a policy for the Rover which
        results in a trajectory that moves the rover from
        start point to end point while avoiding the obstacles,
        thereby maximizing reward 
    ''' 
    def __init__(
        self,
        xs_to_scores_dict={},
        num_calls=0,
        dim=60,
        tau=None,
        **kwargs,
    ):
        assert dim % 2 == 0
        lb = -0.5 * 4 / dim 
        ub = 4 / dim 

        # create rover domain 
        self.domain = create_large_domain(n_points=dim // 2)
        # create rover oracle 
        f_max=5.0 # default
        self.oracle = ConstantOffsetFn(self.domain, f_max)
        # create distance module for divf
        self.dist_module = Distance()
        # create dict to hold mapping from points to trajectories 
        self.xs_to_trajectories_dict = {}
        # rover oracle needs torch.double datatype 
        self.tkwargs={"dtype": torch.double}

        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict,
            num_calls=num_calls,
            task_id='rover',
            dim=dim,
            lb=lb,
            ub=ub,
            **kwargs,
        ) 


    def query_oracle(self, x):
        reward = torch.tensor(self.oracle(x.cpu().numpy())).to(**self.tkwargs) # .unsqueeze(-1)
        return reward 
    

    def get_trajectory(self, x):
        try:
            trajectory = self.xs_to_trajectories_dict[x]
        except:
            trajectory = torch.from_numpy(self.domain.trajectory(x.cpu().numpy())).to(**self.tkwargs)
            self.xs_to_trajectories_dict[x] = trajectory

        return trajectory


    def divf(self, x1, x2 ):
        traj1 = self.get_trajectory(x1)
        traj2 = self.get_trajectory(x2)

        return self.get_one_way_distance(traj1, traj2)


    def get_one_way_distance(self, trajA, trajB): 
        ''' Returns one way distance (OWD) between two trajectories
            (https://zheng-kai.com/paper/vldbj_2019.pdf)
                d_proj(A,B) = (1/N_points) * Sum of Euclidian dist from each point in the A, to NEAREST point in the B 
                d(A, B) = mean(d_proj(A, B), d_proj(B, A)) --> symmetric distance metric 
            Also supports batches of pariwise trajectories, 
                (will output distances between each pair in batch)
        '''
        N = trajA.shape[-2]
        trajA, trajB = trajA.cuda(), trajB.cuda() # each bsz x 1000 x 2 
        dist_matrix = self.dist_module._dist(trajA, trajB, postprocess=False) # bsz x 1000, 1000 
        dists_AB, _ = torch.min(dist_matrix, dim=-1)
        dists_BA, _ = torch.min(dist_matrix, dim=-2) 

        return 0.5*( (dists_AB.sum(dim=-1)/N)+(dists_BA.sum(dim=-1)/N) ).detach().cpu() 

