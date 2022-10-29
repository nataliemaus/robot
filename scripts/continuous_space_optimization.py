import sys
sys.path.append("../")
import fire
import torch 
from scripts.optimize import Optimize
from robot.tasks.lunar_lander.lunar_lander_objective import LunarLanderObjective
from robot.tasks.rover.rover_objective import RoverObjective
from robot.tasks.stocks.stocks_objective import StocksObjective

task_id_to_objective = {}
task_id_to_objective['lunar'] = LunarLanderObjective
task_id_to_objective['rover'] = RoverObjective
task_id_to_objective['stocks'] = StocksObjective

class ContinuousSpaceOptimization(Optimize):
    """
    Run ROBOT Optimization on Continuous Space Example Tasks 
        (ie NOT a structured task --> does NOT require a VAE and LolRobot)
    """
    def __init__(
        self,
        num_initialization_points=1024,
        **kwargs,
    ):
        super().__init__(
            num_initialization_points=num_initialization_points,
            **kwargs
        )

        # add args to method args dict to be logged by wandb
        self.method_args['opt'] = locals()
        del self.method_args['opt']['self']


    def initialize_objective(self):
        self.objective = task_id_to_objective[self.task_id](
            tau=self.tau
        )

        return self


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Defines the following:
                self.init_train_x (a tensor of x's)
                self.init_train_y (a tensor of scores/y's)
        '''
        lb, ub = self.objective.lb, self.objective.ub 
        xs = torch.rand(self.num_initialization_points, self.objective.dim)*(ub - lb) + lb
        out_dict = self.objective(xs)
        self.init_train_x = torch.from_numpy(out_dict['valid_xs']).float() 
        self.init_train_y = torch.tensor(out_dict['scores']).float() 
        self.init_train_y = self.init_train_y.unsqueeze(-1)
        
        if self.verbose:
            print("Loaded initial training data")
            print("train x shape:", self.init_train_x.shape)
            print("train y shape:", self.init_train_y.shape)

        return self 


if __name__ == "__main__":
    fire.Fire(ContinuousSpaceOptimization)
