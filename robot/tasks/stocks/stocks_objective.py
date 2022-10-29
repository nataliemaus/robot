import torch 
from robot.objective import Objective
import pandas as pd 


class StocksObjective(Objective):
    '''S and P 500 optimization task
        Goal is to maximize Sharpe Value of Portfolio
        Computed over the past 3 years of the S and P 500 
        (2019-01-01 to 2021-12-31)
    ''' 
    def __init__(
        self,
        tau,
        xs_to_scores_dict={},
        num_calls=0,
        **kwargs,
    ):
        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict,
            num_calls=num_calls,
            task_id='stocks',
            dim=500,
            lb=0.0,
            ub=1.0,
            **kwargs,
        )
        portfolio = pd.read_csv('../robot/tasks/stocks/SP500_3years.csv')  # (756, 500)
        assert int(portfolio.shape[1]) == 500 # check that all 500 stocks were loaded
        # self.stocks_list = self.portfolio.keys().tolist()
        self.portfolio_tensor = torch.tensor(portfolio.values).float() 
        self.tau = tau


    def query_oracle(self, weights):
        if not torch.is_tensor(weights):
            weights = torch.tensor(weights).float() 
        weights = weights.cuda() 
        # Normalize weights so they sum to 1.0 
        # (We always invest 100% of available funds) 
        norm = torch.linalg.norm(weights, ord=1)  
        if norm == 0:
            norm = 1e-10 
        weights = weights/norm 
        # Reward_i = (Endprice/ Startprice) * Weight_i 
        # Ri_dayj = (Dayj_price/startprice) * Weight_i 
        start_prices = self.portfolio_tensor[0,:].cuda() # 756 x 500 --> 500, 
        daily_price_changes = self.portfolio_tensor.cuda()/(start_prices.reshape(1,500) ) # (756, 500)
        total_daily_rewards = daily_price_changes @ weights # (756,)
        assert len(weights) == self.dim 
        assert weights.ndim == 1 
        assert torch.isclose(weights.sum(), torch.tensor(1.0)) # normalization should guarantee this 
        # R = weighted total return over 3 years 
        R = total_daily_rewards[-1] 
        # S = Standard Deviation on Daily return changes (Volatility)
        all_day_to_day_changes = torch.diff(total_daily_rewards, n=1) 
        S = torch.std(all_day_to_day_changes) 
        # 252 trading days per year (normalization constant)
        # Here we assume a risk-free-rate of 0 (Sharpe Ratio = (R - rf / S) where rf is risk free return)
        sharpe_ratio = R/(S*torch.sqrt(torch.tensor(252)))

        return sharpe_ratio.item() 


    def divf(
        self,
        weights1,
        weights2,
        stop_at_minDist=True,
    ):
        ''' Portfolio TopK Distance:
        Distance between two stock portfolios = 
        Maximum value k such that top k stocks are all different
        (No intersection between top k stocks for each portfolio)
        ''' 
        if weights1.ndim > 1:
            weights1 = weights1.squeeze() 
        if weights2.ndim > 1:
            weights2 = weights2.squeeze() 
        assert len(weights1) == 500 
        assert weights1.ndim == 1
        assert len(weights2) == 500 
        assert weights2.ndim == 1
        _, ordered_idxs1 = torch.topk(weights1, len(weights1)) 
        _, ordered_idxs2 = torch.topk(weights2, len(weights1)) 
        for k in range(1,len(ordered_idxs1)): 
            idxs1 = set(ordered_idxs1[0:k].tolist()) 
            idxs2 = set(ordered_idxs2[0:k].tolist())  
            if len(idxs1 & idxs2) > 0: # if top k have any stock in common 
                break 
            # stop computing early when min diversity acheived to save time
            elif stop_at_minDist and (k > self.tau): 
                return k  

        return k - 1 # Max k such that top k stocks are all different 

