import torch
import math
import numpy as np
from robot.trust_region import update_state
from robot.gp_utils.update_models import update_models_end_to_end
from robot.robot import RobotState

class LolRobotState(RobotState):

    def __init__(
        self,
        M,
        tau,
        objective,
        train_x,
        train_y,
        train_z,
        k=1_000,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        acq_func='ts',
        verbose=True,
    ):

        super().__init__(
            M=M,
            tau=tau,
            objective=objective,
            train_x=train_x,
            train_y=train_y,
            train_z=train_z,
            k=k,
            minimize=minimize,
            num_update_epochs=num_update_epochs,
            init_n_epochs=init_n_epochs,
            learning_rte=learning_rte,
            bsz=bsz,
            acq_func=acq_func,
            verbose=verbose,
            )

        self.progress_fails_since_last_e2e = 0
        self.initialize_top_k() # only track top k for LOL-BO, unnessary for regular opt 


    def search_space_data(self):
        return self.train_z


    def randomly_sample_feasible_center(self, higher_ranked_xs, max_n_samples=1_000):
        ''' Rare edge case when we run out of feasible evaluated datapoints
        and must randomly sample to find a new feasible center point
        '''
        n_samples = 0
        while True:
            if n_samples > max_n_samples:
                raise RuntimeError(f'Failed to find a feasible tr center after {n_samples} random samples, recommend tring use of smaller M or smaller tau')
            center_point = self.sample_random_searchspace_points(N=1) 
            center_x = self.objective.vae_decode(center_point)
            n_samples += 1
            if self.is_feasible(center_x, higher_ranked_xs=higher_ranked_xs):
                out_dict = self.objective(center_point, center_x)
                center_score = out_dict['scores'].item() 
                # add new point to existing data 
                self.update_next(
                    z_next=center_point,
                    y_next=torch.tensor(center_score).float(),
                    x_next=center_x,
                )
                break 

        return center_x, center_point, center_score 


    def compute_scores_remaining_cands(self, feasible_searchspace_pts):
        out_dict = self.objective(
            feasible_searchspace_pts,
            self.feasible_xs # pass in to avoid re-decoding the zs to xs
        )
        feasible_xs = out_dict['valid_xs']
        feasible_ys = out_dict['scores']
        feasible_searchspace_pts = out_dict['valid_zs']
        self.all_feasible_xs = self.all_feasible_xs + feasible_xs.tolist() 

        return feasible_ys, feasible_searchspace_pts


    def get_feasible_cands(self, x_cands ):
        self.feasible_xs, bool_arr = self.remove_infeasible_candidates(
            x_cands=x_cands, 
            higher_ranked_cands=self.all_feasible_xs
        )
        feasible_searchspace_pts = self.z_next[bool_arr]
        
        return feasible_searchspace_pts


    def generate_batch_single_tr(self, state):
        self.z_next = super().generate_batch_single_tr(state)
        x_next = self.objective.vae_decode(self.z_next)
        return x_next


    def update_data_all_feasible_points(self,):
        self.update_next(
            z_next_=self.all_feasible_searchspace_pts,
            y_next_=torch.tensor(self.all_feasible_ys).float(),
            x_next_=self.all_feasible_xs,
            acquisition=True
        )


    def initialize_top_k(self):
        ''' Initialize top k x, y, and zs'''
        # track top k scores found
        self.top_k_scores, top_k_idxs = torch.topk(self.train_y.squeeze(), min(self.k, len(self.train_y)))
        self.top_k_scores = self.top_k_scores.tolist()
        top_k_idxs = top_k_idxs.tolist()
        self.top_k_xs = [self.train_x[i] for i in top_k_idxs]
        self.top_k_zs = [self.train_z[i].unsqueeze(-2) for i in top_k_idxs]


    def update_next(self, z_next_, y_next_, x_next_, acquisition=False):
        '''Add new points (z_next, y_next, x_next) to train data
            and update progress (top k scores found so far)
        '''
        # if no progess made on acqusition, count as a failure
        if (len(x_next_) == 0) and acquisition:
            self.progress_fails_since_last_e2e += 1
            return None 

        z_next_ = z_next_.detach().cpu() 
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze() 
        progress = False
        for i, score in enumerate(y_next_):
            self.train_x.append(x_next_[i] )
            if len(self.top_k_scores) < self.k: 
                # if we don't yet have k top scores, add it to the list
                self.top_k_scores.append(score.item())
                self.top_k_xs.append(x_next_[i])
                self.top_k_zs.append(z_next_[i].unsqueeze(-2))
            elif score.item() > min(self.top_k_scores) and (x_next_[i] not in self.top_k_xs):
                # if the score is better than the worst score in the top k list, upate the list
                min_score = min(self.top_k_scores)
                min_idx = self.top_k_scores.index(min_score)
                self.top_k_scores[min_idx] = score.item()
                self.top_k_xs[min_idx] = x_next_[i]
                self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2) # .cuda()
            #if we imporve 
            if score.item() > self.best_score_seen:
                self.progress_fails_since_last_e2e = 0
                progress = True
                self.best_score_seen = score.item() 
                self.best_x_seen = x_next_[i]
        if (not progress) and acquisition: # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        self.train_y = torch.cat((self.train_y, y_next_), dim=-2)
        self.train_z = torch.cat((self.train_z, z_next_), dim=-2)


    def update_models_e2e(self):
        '''Finetune VAE end to end with surrogate model
        '''
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.num_new_points:]
        new_ys = self.train_y[-self.num_new_points:].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()
        self.objective, self.model = update_models_end_to_end(
            train_x,
            train_y,
            self.objective,
            self.model,
            self.mll,
            self.learning_rte,
            self.num_update_epochs
        )

        # As in LOL-BO, after the after e2e update, 
        #   we recenter by passing points back throough VAE 
        #   to find new locations in fine-tuned latent space 
        self.recenter_vae()


    def recenter_vae(self):
        '''Pass SELFIES strings back through
            VAE to find new locations in the
            new fine-tuned latent space
        '''
        self.objective.vae.eval()
        self.model.train()
        optimizer1 = torch.optim.Adam([{'params': self.model.parameters(),'lr': self.learning_rte} ], lr=self.learning_rte)
        new_xs = self.train_x[-self.num_new_points:]
        train_x = new_xs + self.top_k_xs
        max_string_len = len(max(train_x, key=len))
        # max batch size smaller to avoid memory limit 
        #   with longer strings (more tokens) 
        bsz = max(1, int(2560/max_string_len))
        num_batches = math.ceil(len(train_x) / bsz) 
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
                batch_list = train_x[start_idx:stop_idx] 
                z, _ = self.objective.vae_forward(batch_list)
                out_dict = self.objective(z)
                scores_arr = out_dict['scores'] 
                valid_zs = out_dict['valid_zs']
                selfies_list = out_dict['valid_xs']
                if len(scores_arr) > 0: # if some valid scores
                    scores_arr = torch.from_numpy(scores_arr)
                    if self.minimize:
                        scores_arr = scores_arr * -1
                    pred = self.model(valid_zs)
                    loss = -self.mll(pred, scores_arr.cuda())
                    optimizer1.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer1.step() 
                    with torch.no_grad(): 
                        z = z.detach().cpu()
                        self.update_next(z,scores_arr,selfies_list)
            torch.cuda.empty_cache()
        self.model.eval() 

