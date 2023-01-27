#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from gp import train_gp
from utils import from_unit_cube, latin_hypercube, to_unit_cube, centroid_x, euclidean_distance
import torch
from torch.quasirandom import SobolEngine
from copy import deepcopy, copy
from scipy.stats import gmean


class RaboTS:
    """The RABO-TS algorithm.
    """
    
    def __init__(self, dim, x_lb, x_ub, n_init=0, n_cand=0, batch_size=1, is_minimisation=False):
        
        self.is_minimisation = is_minimisation  # is the bo problem a maximiation or minimisation problem
        self.dim = dim
        self.x_lb = x_lb  #-1 * np.ones(dim)
        self.x_ub = x_ub  #2 * np.ones(dim)
        
        self.X = np.zeros((0, self.dim))  # [ [,,] [,,] ...]
        self.fX = np.zeros((0, 1))  # [ [] [] ...]
        
        self.n_init = n_init
        
        # number of candidate to use in each iteration
        if (n_cand == 0):
            self.num_cand =  min(100 * self.dim, 5000) #100
        else:
            self.num_cand = n_cand
        
        self.ls_rw_idx = []
        self.ls_av_idx = []
        
        self.use_global_gp = False
        self.batch_size = batch_size
        
        
        self.tr_radius_init = 1.0
        self.tr_radius = self.tr_radius_init #1.0
        self.tr_radius_final  = np.ones((1, self.dim))
        self.tr_x_center = np.empty((0, self.dim))
        
        self.tr_lb = 0 * np.ones((1, self.dim))
        self.tr_ub = 1 * np.ones((1, self.dim))
        self.tr_lb_2 = 0 * np.ones((1, self.dim))
        self.tr_ub_2 = 1 * np.ones((1, self.dim))
        self.has_tr_2 = False
        
        self.ls_from_tr2 = [False] * self.batch_size
        
        self.no_improve_rw_contract_th = self.dim * 1 +  self.batch_size * 1
        self.num_iter_no_improve = 0
        
        
        # Figure out what device we are running on
        if (False): #torch.cuda.is_available(): #self.dim < 5:
            self.device, self.dtype = torch.device("cuda"), torch.float32
            print("torch cuda available")
        else:
            self.device, self.dtype = torch.device("cpu"), torch.float64
            print("torch cuda not available")
            #assert torch.cuda.is_available(), "can't use cuda if it's not available"
        
        
    def observe(self, X, fX, update_tr=True):
        """Feed an observation back.
        
        Parameters
        ----------
        X : np array-like, shape (n,[dim])
            Places where the objective function has already been evaluated.
        y : np array-like, shape (n,1)
            Corresponding values where objective has been evaluated

        """
        
        # add the data to the dataset
        self.X = np.vstack((self.X, X))
        self.fX = np.vstack((self.fX, fX))
    
        if (len(self.fX) >= self.n_init):
            self._partition()
            self._construct_gp()
            if (len(self.fX) >= self.n_init + 1):
                if (update_tr):
                    self._adjust_trust_region()
          
        
        
    
    def _construct_gp(self):
        
        """ Fit GPs rw & av. """
        
        # fit rw GP
        rw_X = deepcopy(self.X[self.ls_rw_idx, :])
        rw_X = to_unit_cube(rw_X, self.x_lb, self.x_ub)
        rw_fX = deepcopy(self.fX[self.ls_rw_idx, :])
        self.rw_fX_mean = np.mean(rw_fX)
        self.rw_fX_std = np.std(rw_fX)
        #self.rw_fX_std = 1.0 if rw_fX_std < 1e-6 else rw_fX_std
        if (self.rw_fX_std < 0.0001):
            self.rw_fX_std = 0.0001
        rw_fX_norm = (rw_fX - self.rw_fX_mean) / self.rw_fX_std
        
        rw_X_torch = torch.tensor(rw_X).to(device=self.device, dtype=self.dtype)
        rw_y_torch = torch.tensor(np.ravel(rw_fX_norm)).to(device=self.device, dtype=self.dtype)
        self.rw_gp = train_gp(train_x=rw_X_torch, train_y=rw_y_torch, use_ard=True, num_steps=50)
        
        rw_gp_noise = self.rw_gp.likelihood.noise.cpu().detach().numpy().ravel()
        rw_gp_outputscale = self.rw_gp.covar_module.outputscale.cpu().detach().numpy().ravel()
        rw_gp_lengthscale = self.rw_gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        self.rw_gp_lengthscale = copy(rw_gp_lengthscale)
        
        # fit av GP
        av_X = deepcopy(self.X[self.ls_av_idx, :])
        av_X = to_unit_cube(av_X, self.x_lb, self.x_ub)
        av_fX = deepcopy(self.fX[self.ls_av_idx, :])
        self.av_fX_mean = np.mean(av_fX)
        self.av_fX_std = np.std(av_fX)
        if (self.av_fX_std < 0.0001):
            self.av_fX_std = 0.0001
        #self.av_fX_std = 1.0 if av_fX_std < 1e-6 else av_fX_std
        av_fX_norm = (av_fX - self.av_fX_mean) / self.av_fX_std
        
        av_X_torch = torch.tensor(av_X).to(device=self.device, dtype=self.dtype)
        av_y_torch = torch.tensor(np.ravel(av_fX_norm)).to(device=self.device, dtype=self.dtype)
        self.av_gp = train_gp(train_x=av_X_torch, train_y=av_y_torch, use_ard=True, num_steps=50)
        
        del rw_X_torch, rw_y_torch, av_X_torch, av_y_torch
        
        
        """ Marix of X covariance to be used for calculating weights. """
        # calculate weights - distance from best_X and worst_X
        len_half_ls_rw_idx = max(1, int(len(self.ls_rw_idx) * 0.5))
        len_half_ls_av_idx = max(1, int(len(self.ls_av_idx) * 0.5))
        print("len_half_ls_rw_idx: ", len_half_ls_rw_idx, ", len_half_ls_av_idx: ", len_half_ls_av_idx)
        self.rw_X_torch_covar = torch.tensor(rw_X[:len_half_ls_rw_idx, :])
        self.av_X_torch_covar = torch.tensor(av_X[-len_half_ls_av_idx:, :])
        
        
        
        
        
        
    def suggest(self, X_cand = None, seed=None):
        
        
        
        """ Calculate num_cand for each trust region based on geometry mean """
        if self.has_tr_2 :
            num_cand = [0] * (self.dim + 1)
            geo_mean = [0] * (self.dim + 1)
            tr_len = [0] * (self.dim + 1)
            
            # geometry mean of main tr
            tr_len[0] = np.ravel(self.tr_radius_final) * 2
            geo_mean[0] = gmean(tr_len[0])
            
            # geometry mean of each sub tr
            for i in range(self.dim):
                tr_lb_2_ = np.ravel(self.tr_lb_2[i,:])
                tr_ub_2_ = np.ravel(self.tr_ub_2[i,:])
                tr_len[i+1] = tr_ub_2_ - tr_lb_2_
                geo_mean[i+1] = gmean(tr_len[i+1])
             
            geo_mean_count = 0
            for m in geo_mean:
                if m > 0: geo_mean_count += 1
                
            if (geo_mean_count <= 1):
                # only one tr, all auxiliary tr are not valid
                num_cand = [self.num_cand]
            else:
                num_cand_coefficient = 0.5 
                
                for i in range(len(num_cand)):
                    if (i == 0):
                        num_cand[i] = int( self.num_cand  / geo_mean_count )
                    else:
                        if (geo_mean[i] > 0):
                            num_cand[i] = int( self.num_cand / geo_mean_count )
                     
        else:
            num_cand = [self.num_cand]
        
        
        """Draw X candidates."""
        # Draw a Sobolev sequence in [lb, ub]
        if (seed == None):
            seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        X_cand_sobol = sobol.draw(num_cand[0]).detach().numpy()
        
        
        # At the first iteration, tr_x_center is empty, use the best x as tr_x_center
        if (self.tr_x_center.size == 0):
            self.tr_x_center = self.X[self.fX.argmin().item(), :][None, :]
            self.tr_x_center  = to_unit_cube(self.tr_x_center , self.x_lb, self.x_ub)
            self.tr_x_center_fX = self.fX.min()  # np.min(self.fX[:len_])
            
        
        
        # Draw x candidates from the main tr
        X_cand_sobol_rw = self.tr_lb + (self.tr_ub - self.tr_lb) * X_cand_sobol  # this convert samples from sobol to within ub & lb
        X_cand_rw = self.tr_x_center.copy() * np.ones((num_cand[0], self.dim))
        X_cand_rw[:, :] = X_cand_sobol_rw[:, :]
        
        if (self.dim == 1):
            # sorting is just for the purpose of plotting
            X_cand_rw = np.sort(X_cand_rw, axis=None)     # sort the flattened array, this returns [], so need to reshape
            X_cand_rw = X_cand_rw.reshape(len(X_cand_rw), self.dim)
            
        
        X_cand = deepcopy(X_cand_rw)
        
        # Draw x candidates from the auxiliary tr
        if (self.has_tr_2):
            # geometry mean of each sub tr
            for i in range(self.dim):
                tr_lb_2_ = np.ravel(self.tr_lb_2[i,:])
                tr_ub_2_ = np.ravel(self.tr_ub_2[i,:])
                valid_tr = True
                
                for j in range(self.dim):
                    if (tr_lb_2_[j] < tr_ub_2_[j]):
                        pass
                    else:
                        valid_tr = False
                        break
                
                if (valid_tr):
                    if (num_cand[i+1] > 0):
                        X_cand_2_sobol = tr_lb_2_ + (tr_ub_2_ - tr_lb_2_) * X_cand_sobol  # this convert samples from sobol to within ub & lb
                        X_cand_2 = self.tr_x_center_flip_rw.copy() * np.ones((num_cand[i+1], self.dim))
                        X_cand_2[:, :] = X_cand_2_sobol[:, :]
                        X_cand = np.vstack((X_cand, deepcopy(X_cand_2)))
            
        self.X_cand = deepcopy(X_cand)
        
        
        """ Predict y candidates with GPs. """
        #X_cand_torch = torch.tensor(X_cand)
        X_cand_torch = torch.tensor(X_cand).to(device=self.device, dtype=self.dtype)
        X_cand_torch = X_cand_torch.double() # ExactGP forward function requires double (float64)
        
        # predict y using rw gp 
        rw_y_cand = self.rw_gp.likelihood(self.rw_gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()
        rw_y_cand_pred = self.rw_gp(X_cand_torch)
        rw_y_cand_pred_mean = rw_y_cand_pred.mean.detach().numpy()
        rw_y_cand_pred_stdev = rw_y_cand_pred.stddev.detach().numpy()
        
        # predict y using av gp 
        av_y_cand = self.av_gp.likelihood(self.av_gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()
        av_y_cand_pred = self.av_gp(X_cand_torch)
        av_y_cand_pred_mean = av_y_cand_pred.mean.detach().numpy()
        av_y_cand_pred_stdev = av_y_cand_pred.stddev.detach().numpy()

        # De-standardize the mean values
        #rw_y_cand_pred_mean = self.rw_fX_mean + self.rw_fX_std * rw_y_cand_pred_mean
        #av_y_cand_pred_mean = self.av_fX_mean + self.av_fX_std * av_y_cand_pred_mean
        
        # De-standardize the sampled values
        rw_y_cand_norm = deepcopy(rw_y_cand)
        #rw_y_cand = self.rw_fX_mean + self.rw_fX_std * rw_y_cand
        
        av_y_cand_norm = deepcopy(av_y_cand)
        #av_y_cand = self.av_fX_mean + self.av_fX_std * av_y_cand
        
        
        """ Calculate weights. """
        # calculate weights - distance from best_X and worst_X
        
        rw_covar_x = self.rw_gp.covar_module(X_cand_torch, self.rw_X_torch_covar) # this returns matrix X_cand_len X rw_X_len
        av_covar_x = self.av_gp.covar_module(X_cand_torch, self.av_X_torch_covar)
        y_cand = deepcopy(rw_y_cand)
        y_cand_pred_mean = deepcopy(rw_y_cand_pred_mean)
        for i in range(len(X_cand)):
            rw_mean = np.mean(np.array(rw_covar_x[i,:].detach().numpy().ravel()))  # avg distance to all rw points
            av_mean = np.mean(np.array(av_covar_x[i, :].detach().numpy().ravel())) # avg distance to all av points
            sum_ = rw_mean + av_mean
            rw_weight = rw_mean / sum_
            av_weight = av_mean / sum_
            
            if (True):#(abs(rw_weight - av_weight) < 0.1):
                y_cand[i, :] = (rw_y_cand[i, :] * rw_weight) + (av_y_cand[i, :] * av_weight )
                
            elif (rw_weight > av_weight):
                #y_cand[i, :]  = rw_y_cand[i, :] * (rw_weight**(av_weight/1)) 
                y_cand[i, :]  = (rw_y_cand[i, :] * (rw_weight**(av_weight/1)) ) + (av_y_cand[i, :] * (av_weight**(1 + (rw_weight*1) )))
            else:
                #y_cand[i, :]  = av_y_cand[i, :]  * (av_weight**(rw_weight/1 ))
                y_cand[i, :]  = (rw_y_cand[i, :] * (rw_weight**(1 + (av_weight*1))) ) + (av_y_cand[i, :] * (av_weight**(rw_weight/1 )))
                
            y_cand_pred_mean[i] = deepcopy(y_cand[i, 0]  )
        
        acq_scores = deepcopy(y_cand)  # y_cand is de-standardise fX
        acq_scores_ = deepcopy(acq_scores)
        rw_acq_scores_ = deepcopy(rw_y_cand)
        av_acq_scores_ = deepcopy(av_y_cand)
        
        
        
        """ Select candidates of batch size. """
        X_next = np.ones((self.batch_size, self.dim))
        y_next = np.ones((self.batch_size, 1))
        self.ls_from_tr2 = [False] * self.batch_size
        for i in range(self.batch_size):
            
            # Pick the best point and make sure we never pick it again
            if ((self.has_tr_2)) and (num_cand[0] < self.num_cand) and (self.batch_size > 1):
                # pick half of the best points from tr 0 and another half from other sub tr
                num_from_tr2 = max(1, int(self.batch_size * (num_cand_coefficient)))
                if (i < num_from_tr2):
                    len_ = len(acq_scores) - num_cand[0]
                    indbest = num_cand[0] + np.argmin(acq_scores[-len_:, i])
                    assert acq_scores[-len_:, i].min() == acq_scores[indbest, i]
                    self.ls_from_tr2[i] = True
                else:
                    indbest = np.argmin(acq_scores[:num_cand[0], i])    # min of the first num_cand[0] candidates
                    assert acq_scores[:num_cand[0], i].min() == acq_scores[indbest, i]
                
            else:
                indbest = np.argmin(acq_scores[:, i])
                assert acq_scores[:, i].min() == acq_scores[indbest, i]
            
            # is the candidate from new trust region 
            if (indbest > num_cand[0]):
                self.ls_from_tr2[i] = True
            
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_next[i, :] = deepcopy(y_cand[indbest, i])
            #y_next[i, :] = deepcopy(y_cand_pred_mean[indbest])
            
            acq_scores[indbest, :] = np.inf
            
        
        self.pred_fX_next = deepcopy(y_next)
        #print("pred_fX_next: ", self.pred_fX_next)
        
        # Undo the warping
        X_cand = from_unit_cube(X_cand, self.x_lb, self.x_ub)
        X_next = from_unit_cube(X_next, self.x_lb, self.x_ub)
        x_best_cand = X_next
        y_best_cand = y_next
        
        self.X_next = deepcopy(X_next)
        
        
        return x_best_cand, y_best_cand, X_cand, rw_y_cand, rw_y_cand_norm, rw_y_cand_pred_mean, rw_y_cand_pred_stdev, \
                av_y_cand, av_y_cand_norm, av_y_cand_pred_mean, av_y_cand_pred_stdev, y_cand_pred_mean, acq_scores_, rw_acq_scores_, av_acq_scores_
        
      
        
      
    def _partition(self):
        
        ls_fX = np.ravel(self.fX)
        ls_fX_sort = np.sort(ls_fX)  # sort from low to high
        ls_fX_argsort = np.argsort(ls_fX)
        
        self.ls_rw_idx = []
        self.ls_av_idx = []
        
        ctr_rw_added = 0
        ctr_av_added = 0
        if (self.is_minimisation):
            self.best_fX = ls_fX_sort[0]
            #self.best_fX_prev = ls_fX_sort[1]
            self.worst_fX = ls_fX_sort[-1]
            for i in range(len(ls_fX_sort)):
                idx = ls_fX_argsort[i]
                dist_to_best_fX = abs(ls_fX_sort[i] - self.best_fX)
                dist_to_worst_fX = abs(ls_fX_sort[i] - self.worst_fX)
                if (dist_to_best_fX < dist_to_worst_fX):
                    if (not (idx in self.ls_rw_idx)) :
                        self.ls_rw_idx.append(idx)
                        ctr_rw_added += 1
                else:
                    if (not (idx in self.ls_av_idx)) :
                        self.ls_av_idx.append(idx)
                        ctr_av_added += 1
         
        if (len(self.ls_rw_idx) <= self.dim + 1):
            num_to_add = min(int(len(self.ls_av_idx)/2), self.dim + 1 - len(self.ls_rw_idx))
            for i in range(num_to_add):
                self.ls_rw_idx.append(self.ls_av_idx[i])
            
          
        if (len(self.ls_av_idx) <= self.dim + 1):
            num_to_add = min(int(len(self.ls_rw_idx)/2), self.dim + 1 - len(self.ls_av_idx))
            for i in range(len(self.ls_rw_idx)-1, -1, -1):
                idx = self.ls_rw_idx[i]
                if (not (idx in self.ls_av_idx)) :
                    self.ls_av_idx.insert(0, idx)
                if (i >= num_to_add): break 
            
                
        self.use_global_gp = False
        if (len(self.ls_rw_idx) <= 1) or (len(self.ls_av_idx) <= 1):
            self.use_global_gp = True
        
        
        
        best_X_idx = self.fX.argmin().item()
        best_X = self.X[best_X_idx, :][None, :]
        self.best_X = to_unit_cube(best_X, self.x_lb, self.x_ub)
        
        
        worst_X_idx = self.fX.argmax().item()
        worst_X = self.X[worst_X_idx, :][None, :]
        self.worst_X = to_unit_cube(worst_X, self.x_lb, self.x_ub)
        
        
        
    """ Predict fX using current rw & av gp. """
    def _predict_fX(self, x_, verbose=False, fX_type=''):
        
        # prev_best_X, batch_X are in unit cube format
        
        # predict the fX min
        x_min_torch = torch.tensor(np.array(x_)).to(device=self.device, dtype=self.dtype)
        x_min_torch = x_min_torch.double() # ExactGP forward function requires double (float64)
        
        # predict fX min using rw gp 
        rw_fX_min_pred = self.rw_gp(x_min_torch)
        rw_fX_min_pred_mean = rw_fX_min_pred.mean.detach().numpy()
        rw_fX_min_pred_mean_ = copy(rw_fX_min_pred_mean)
        
        # predict fX min using av gp 
        av_fX_min_pred = self.av_gp(x_min_torch)
        av_fX_min_pred_mean = av_fX_min_pred.mean.detach().numpy()
        av_fX_min_pred_mean_ = copy(av_fX_min_pred_mean)
        
        
        
        rw_covar_x_min = self.rw_gp.covar_module(x_min_torch, self.rw_X_torch_covar) # this returns matrix X_cand_len X rw_X_len
        av_covar_x_min = self.av_gp.covar_module(x_min_torch, self.av_X_torch_covar)
        rw_mean_ = np.mean(np.array(rw_covar_x_min[0,:].detach().numpy().ravel()))  # avg distance to all rw points
        av_mean_ = np.mean(np.array(av_covar_x_min[0, :].detach().numpy().ravel())) # avg distance to all av points
        sum_ = rw_mean_ + av_mean_
        rw_weight = rw_mean_ / sum_
        av_weight = av_mean_ / sum_
        
        if (verbose):
            print("rw_fX_min_pred_mean: ", rw_fX_min_pred_mean, ", av_fX_min_pred_mean: ", av_fX_min_pred_mean)
            
            
        pred_fX_min = deepcopy(rw_fX_min_pred_mean)
        if (fX_type == 'rw'):
            pred_fX_min[0]  = rw_fX_min_pred_mean[0]
        elif (fX_type == 'av'):
            pred_fX_min[0]  = av_fX_min_pred_mean[0]
        else:
        
            pred_fX_min = deepcopy(rw_fX_min_pred_mean)
            if (True):#(abs(rw_weight - av_weight) < 0.1):
                pred_fX_min[0] = (rw_fX_min_pred_mean[0] * rw_weight + av_fX_min_pred_mean[0] * av_weight ) 
            elif (rw_weight > av_weight):
                #pred_fX_min[0]  = rw_fX_min_pred_mean[0] * (rw_weight**(av_weight/1))
                pred_fX_min[0]  = (rw_fX_min_pred_mean[0] * (rw_weight**(av_weight/1)) ) + (av_fX_min_pred_mean[0] * (av_weight**(1 + (rw_weight*1) )))
            else:
                #pred_fX_min[0]  = av_fX_min_pred_mean[0] * (av_weight**(rw_weight/1 ))
                pred_fX_min[0]  = (rw_fX_min_pred_mean[0] * (rw_weight**(1 + (av_weight*1))) ) + (av_fX_min_pred_mean[0] * (av_weight**(rw_weight/1 )))
        
        
        # Remove the torch variables
        del x_min_torch
        
        pred_fX = copy(pred_fX_min)
        return pred_fX, rw_weight, av_weight
        
    
    
    """ Adjust tr_radius, tr_center, tr lb & ub. """
    def _adjust_trust_region(self):
        
        new_min_fX_found = False
        has_shrink_within = False
        self.has_tr_2 = False
        is_new_min_from_tr2 = False
        ls_diff_fX_last_best = []
        ls_pred_diff_fX_last_best = []
        shrink_within_ctr = 0
        ls_new_min_diff_fX_last_best = []
        ls_new_min_pred_diff_fX_last_best = []
        new_min_ctr = 0
        
        #fX_last = self.fX[-1:]
        batch_fX_ = deepcopy(self.fX[-self.batch_size:])
        batch_fX = np.ravel(batch_fX_)
        batch_fX_arg = [i for i in range(self.batch_size)]
        #batch_fX = np.sort(np.ravel(batch_fX_))[::-1]  # sort from high to low
        #batch_fX_arg = np.argsort(np.ravel(batch_fX_))[::-1]   #  this is the argument of the batch_fX - [0 3 4 2 1]
        
        batch_X = deepcopy(self.X[-self.batch_size:])
        batch_X = to_unit_cube(batch_X, self.x_lb, self.x_ub)
        
        
        # the best fX before receiving the new batch of fX
        len_ = len(self.fX) - self.batch_size
        prev_best_fX = np.min(self.fX[:len_])
        prev_worst_fX = np.max(self.fX[:len_])
        
        
        prev_best_X_idx = np.argmin(self.fX[:len_]) #self.fX.argmin().item()
        prev_best_X = deepcopy(self.X[prev_best_X_idx, :][None, :])
        prev_best_X = to_unit_cube(prev_best_X, self.x_lb, self.x_ub)
        
        batch_fX_best = np.min(batch_fX)
        if (batch_fX_best < prev_best_fX):
            new_min_fX_found = True
        
        
        # actual reduction
        #ls_actual_fX_reduction = []
        actual_diff_fX_min_max = self.worst_fX - self.best_fX
        assert actual_diff_fX_min_max > 0
        ls_actual_fX_reduction = (prev_best_fX - batch_fX) #/ actual_diff_fX_min_max
        
        
        # predicted reduction
        pred_fX_min, rw_weight, av_weight = self._predict_fX(self.best_X, verbose=True, fX_type='rw')
        pred_fX_max, rw_weight, av_weight = self._predict_fX(self.worst_X, verbose=True, fX_type='av')
        pred_prev_fX_min, rw_weight, av_weight = self._predict_fX(prev_best_X, verbose=True, fX_type='rw')
        pred_batch_fX = copy(batch_fX)
        for i in range(len(batch_X)): 
            pred_batch_fX[i], _, _  = self._predict_fX([batch_X[i]])
           
        pred_diff_fX_min_max = pred_fX_max - pred_fX_min
        assert pred_diff_fX_min_max > 0
        ls_pred_fX_reduction = (pred_prev_fX_min - pred_batch_fX) #/ pred_diff_fX_min_max
        
        # normalise
        ls_fX_actual = []
        ls_fX_actual.append(prev_best_fX)
        for i in range(len(batch_fX)):
            ls_fX_actual.append(batch_fX[i])
        mu = np.mean(ls_fX_actual)
        sigma = np.std(ls_fX_actual)
        if (sigma < 0.0001):
            sigma = 0.0001
        prev_best_fX_norm = (prev_best_fX - mu) / sigma
        batch_fX_norm = (batch_fX - mu) / sigma
        ls_actual_fX_reduction = prev_best_fX_norm - batch_fX_norm
        print("ls_actual_fX_reduction norm: ", ls_actual_fX_reduction)
            
        ls_fX_pred = []
        ls_fX_pred.append(pred_prev_fX_min)
        for i in range(len(pred_batch_fX)):
            ls_fX_pred.append(pred_batch_fX[i])
        mu = np.mean(ls_fX_pred)
        sigma = np.std(ls_fX_pred)
        if (sigma < 0.0001):
            sigma = 0.0001
        pred_prev_fX_min_norm = (pred_prev_fX_min - mu) / sigma
        pred_batch_fX_norm = (pred_batch_fX - mu) / sigma
        ls_pred_fX_reduction = pred_prev_fX_min_norm - pred_batch_fX_norm 
        print("ls_pred_fX_reduction norm: ", ls_pred_fX_reduction)
            
        
    
        """ Analyse fX collected in batch. """
        
        for i in range(len(batch_fX)): 
            
            fX_last = deepcopy(batch_fX[i] )#fX
            fX_idx = len(self.fX) - self.batch_size + batch_fX_arg[i]
            
            diff_fX_last_best = (prev_best_fX - fX_last)
            
            if (fX_last < prev_best_fX):
                print("* new min!")
                ls_new_min_diff_fX_last_best.append(copy((ls_actual_fX_reduction[i])))
                ls_new_min_pred_diff_fX_last_best.append(copy(ls_pred_fX_reduction[i]))
                new_min_fX_found = True
                new_min_ctr += 1
                if (self.ls_from_tr2[i]): 
                    is_new_min_from_tr2 = True
                    print(" ***** is_new_min_from_tr2 !!!")
                self.num_iter_no_improve = 0
            
            elif ((diff_fX_last_best) < 0) and (self.ls_from_tr2[i] == False): # (fX_last < self.rw_fX_max):#delta_shrink): 
                if (self.tr_radius > 0.01): 
                    ls_diff_fX_last_best.append(copy((ls_actual_fX_reduction[i])))
                    ls_pred_diff_fX_last_best.append(copy(ls_pred_fX_reduction[i]))
                    has_shrink_within = True
                    shrink_within_ctr += 1
 
            
        if (new_min_fX_found) and (is_new_min_from_tr2):
            print(" ***** is_new_min_from_tr2 !!! ")
            if (self.tr_radius < (self.tr_radius_init * 0.1)):
                self.tr_radius = self.tr_radius_init * 0.1
                print(" ***** is_new_min_from_tr2 !!! set tr_radius 0.05")
        
        
        """ Compute various centroids & distances. """
        
        best_X_idx = self.fX.argmin().item()
        best_X = self.X[best_X_idx, :][None, :]
        best_X = to_unit_cube(best_X, self.x_lb, self.x_ub)
        
        worst_X_idx = self.fX.argmax().item()
        worst_X = self.X[worst_X_idx, :][None, :]
        worst_X = to_unit_cube(worst_X, self.x_lb, self.x_ub)
        
        rw_X_0 = deepcopy(self.X[self.ls_rw_idx, :])
        rw_X_0 = to_unit_cube(rw_X_0, self.x_lb, self.x_ub)
        centroid_rw_X_0 = centroid_x(rw_X_0, self.dim)
        sum_dist_best_to_centroid_rw_X_0 = np.linalg.norm(best_X - centroid_rw_X_0) #sum(dist_best_to_centroid_rw_X_0) / self.dim
        
        centroid_rw_X = copy(centroid_rw_X_0)
        sum_dist_best_to_centroid_rw_X = copy(sum_dist_best_to_centroid_rw_X_0)
        
        av_X = deepcopy(self.X[self.ls_av_idx, :])
        av_X = to_unit_cube(av_X, self.x_lb, self.x_ub)
        centroid_av_X = centroid_x(av_X, self.dim)
        
   
        """ Adjust tr_radius according to fX collected in batches. """
        
        # how many iteration there has been no improvement 
        self.no_improve_rw_contract_th = max(10, self.dim * 1 + self.batch_size * 1)
        self.no_improve_av_expand_th = self.no_improve_rw_contract_th * 2 #self.dim * 10
        if  (len(self.fX) > self.n_init):
            best_X_idx = self.fX.argmin().item()
            num_iter_no_improve = len(self.fX) - best_X_idx - 1
            self.num_iter_no_improve = num_iter_no_improve
            
            len_ = len(self.fX) - self.batch_size
            prev_best_fX_idx = np.argmin(self.fX[:len_])  
            num_iter_no_improve_bef_new_min = best_X_idx - prev_best_fX_idx
            
        
        radius_to_start = 0.25 
        
        # expand tr_radius when new min found
        if (new_min_fX_found):
            
            self.tr_x_center  = deepcopy(best_X)
            self.stop_shrink = False
            ls_ratio_norm = []
            negative_ctr = 0
            ratio_norm = 0
            th = 0
            
              
            th = 1.0
            expand_th_min = 1.0
            expand_th_max = 1.5
            if (self.tr_radius < radius_to_start):
                th = expand_th_min
                th_add = (expand_th_max - th) * (radius_to_start - self.tr_radius) / (radius_to_start)   # start from 1.0
                th = th + th_add
                
            
            for i in range(len(ls_new_min_diff_fX_last_best)):
                if (True):#(ls_new_min_pred_diff_fX_last_best[i] > 0):
                    ratio = abs(ls_new_min_diff_fX_last_best[i]) / abs(ls_new_min_pred_diff_fX_last_best[i])
                    ratio = max(0, ratio - 0.0)
                    #if (ratio < 0.1): ratio = 0
                    if (False):#(ratio > 1.0):
                        ratio = np.log(1 + ratio)
                    ls_ratio_norm.append(ratio)
                    
                else:
                    ratio = 0.0#1.0
                    ls_ratio_norm.append(ratio)
                    negative_ctr += 1
                    
                if (ls_new_min_pred_diff_fX_last_best[i] < 0):   negative_ctr += 1
            
            if (len(ls_ratio_norm) > 1):
                ls_ratio_norm = ls_ratio_norm / max(ls_ratio_norm)
                
            ratio_norm = np.mean(ls_ratio_norm)
            ratio_norm = np.tanh(ratio_norm) + 1
            fn_ratio = copy(ratio_norm)
            
            
            fn_ratio = np.clip(fn_ratio, expand_th_min, th)
            
            
            self.prev_new_min_fn_ratio = fn_ratio
            self.last_expand_ratio = copy(fn_ratio)
            self.tr_radius = min(self.tr_radius_init, fn_ratio * self.tr_radius)
            
            
            self.prev_new_min_fX_found = True
        
        
        
        
        # shrink tr_radius when no new min is found
        if  (has_shrink_within == True):
            
            fX_ratio = 0
            negative_ctr = 0
            ratio_norm = 0
            ls_ratio_norm = []
            
            
            th = 0.6
            shrink_th_max = 0.9
            
            if (self.tr_radius < radius_to_start):
                #th = 0.7
                th_add_1 = abs(th - shrink_th_max) * (radius_to_start - self.tr_radius) / (radius_to_start)   # start from 0.5
                th_add_2 = 0
                th = min(shrink_th_max, th + th_add_1 + th_add_2)
            
            for i in range(len(ls_diff_fX_last_best)):
                
                if (True):#(ls_pred_diff_fX_last_best[i] > 0):
                    ratio =  abs(ls_diff_fX_last_best[i]) / abs(ls_pred_diff_fX_last_best[i] )
                    ratio = max(0, ratio - 0.0)
                    ls_ratio_norm.append(ratio)
                    
                else:
                    ratio = 0.1
                    ls_ratio_norm.append(ratio)
                    negative_ctr += 1
                    
                if (ls_pred_diff_fX_last_best[i] < 0): negative_ctr += 1
            
            if (len(ls_ratio_norm) > 1):
                ls_ratio_norm = ls_ratio_norm / max(ls_ratio_norm)
                
            ratio_norm = np.mean(ls_ratio_norm)
            ratio_norm = np.exp( -1 * ratio_norm)
            fX_ratio = copy(ratio_norm)
               
            fX_ratio = np.clip(fX_ratio, th, 0.99)  # for rosenbrock 20d which which has fn_ratio 0.9999999
            
            min_tr = 0.01
            self.tr_radius = max(min_tr, fX_ratio * self.tr_radius)
            
            
        
        
        """ Adjust tr_center - contract or expand according to centroid_rw & centroid_av. """
        
        for i in range(self.dim):
            self.tr_radius_final[:, i] = deepcopy(self.tr_radius)
            
        # default tr_center to min x
        self.tr_x_center  = deepcopy(best_X)
        self.tr_x_center_fX = self.fX.min()  # np.min(self.fX[:len_])
        
        if (len(self.fX) > self.n_init + self.batch_size) and (num_iter_no_improve > self.dim * 2):
            tr_x_center_rw = deepcopy(self.tr_x_center)
            
            # contract on centroid rw
            coefficient = 0.0   
            if (sum_dist_best_to_centroid_rw_X < 0.5):
                
                coefficient = min(0.5, (1/ self.dim) * self.tr_radius * (sum_dist_best_to_centroid_rw_X / self.dim) )
                
                for j in range(self.dim):
                    tr_x_center_rw[:,j] = (coefficient * centroid_rw_X[:,j]) +  ( (1 - coefficient) * self.tr_x_center[:,j] )
                
            self.tr_x_center = deepcopy(tr_x_center_rw)
                   
                    
            # expand on centroid rw & centroid av
            if (self.num_iter_no_improve > self.dim * 4 ):
                
                expand_coefficient = 1.0
                tr_x_center_flip_rw = centroid_rw_X.copy()
                tr_x_center_flip_av = centroid_av_X.copy()
                for j in range(self.dim):
                    tr_x_center_flip_rw[:,j] = centroid_rw_X[:,j] + ( (centroid_rw_X[:,j] - best_X[:,j]) * expand_coefficient  ) 
                    tr_x_center_flip_av[:,j] = centroid_av_X[:,j] + ( (centroid_av_X[:,j] - best_X[:,j]) * expand_coefficient )  # move away from best_X
                self.has_tr_2 = True
                self.tr_x_center_flip_rw = tr_x_center_flip_rw.copy()
                self.tr_x_center_flip_av = tr_x_center_flip_av.copy()
                
    
        
        """ Adjust tr lb & ub according to tr_radius & tr_center. """
        # adjust tr lb & ub
    
        weights = 1
        self.tr_lb = self.tr_x_center.copy() * np.ones((1, self.dim))
        self.tr_ub = self.tr_x_center.copy() * np.ones((1, self.dim))
        for i in range(self.dim):
            self.tr_lb[:, i] = self.tr_x_center[:, i] - weights * self.tr_radius_final[:, i]
            self.tr_ub[:, i] = self.tr_x_center[:, i] + weights * self.tr_radius_final[:, i]
        
        self.tr_lb = np.clip(self.tr_lb, 0.0, 1.0)
        self.tr_ub = np.clip(self.tr_ub, 0.0, 1.0)
        
        
        # adjust sub-tr lb & ub
        if (self.has_tr_2 ):
            tr_lb_2_rw = tr_x_center_flip_rw * np.ones((self.dim, self.dim))
            tr_ub_2_rw = tr_x_center_flip_rw * np.ones((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    tr_lb_2_rw[i, j] = 0
                    tr_ub_2_rw[i, j] = 1
                    if (i == j):
                        if (tr_x_center_flip_rw[:,j] > best_X[:,j]):
                            tr_lb_2_rw[i, j] = tr_x_center_flip_rw[:,j]
                        else:
                            tr_ub_2_rw[i, j] = tr_x_center_flip_rw[:,j]
                    
                    else:
                        # remove overlap within all sub tr
                        if (i > 0):
                            if (tr_x_center_flip_rw[:,j] > best_X[:,j]):
                                tr_ub_2_rw[i, j] = tr_x_center_flip_rw[:,j]  #self.tr_lb_2[0, j] # self.tr_x_center_flip[:,j]
                            else:
                                tr_lb_2_rw[i, j] = tr_x_center_flip_rw[:,j]
                    
            
            tr_lb_2_av = tr_x_center_flip_av * np.ones((self.dim, self.dim))
            tr_ub_2_av = tr_x_center_flip_av * np.ones((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    tr_lb_2_av[i, j] = 0
                    tr_ub_2_av[i, j] = 1
                    if (i == j):
                        #if (tr_x_center_flip_av[:,j] > worst_X[:,j]):
                        if (tr_x_center_flip_av[:,j] > best_X[:,j]):
                            tr_lb_2_av[i, j] = tr_x_center_flip_av[:,j]
                        else:
                            tr_ub_2_av[i, j] = tr_x_center_flip_av[:,j]
                    else:
                        # remove overlap within all sub tr
                        if (i > 0):
                            if (tr_x_center_flip_av[:,j] > worst_X[:,j]):
                                tr_ub_2_av[i, j] = tr_x_center_flip_av[:,j]  #self.tr_lb_2[0, j] # self.tr_x_center_flip[:,j]
                            else:
                                tr_lb_2_av[i, j] = tr_x_center_flip_av[:,j]
                    
            
            self.tr_lb_2 = tr_x_center_flip_rw * np.ones((self.dim, self.dim))
            self.tr_ub_2 = tr_x_center_flip_rw * np.ones((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    self.tr_lb_2[i, j] = 0
                    self.tr_ub_2[i, j] = 1
                    if (i == j):
                        if (tr_lb_2_rw[i, j] >= tr_lb_2_av[i, j]) and (tr_ub_2_rw[i, j] <= tr_ub_2_av[i, j]):
                            # tr_rw is within tr_av
                            self.tr_lb_2[i, j] = tr_lb_2_av[i, j] #tr_lb_2_rw[i, j]
                            self.tr_ub_2[i, j] = tr_ub_2_av[i, j] #tr_ub_2_rw[i, j]
                        elif (tr_lb_2_av[i, j] >= tr_lb_2_rw[i, j]) and (tr_ub_2_av[i, j] <= tr_ub_2_rw[i, j]):
                            # tr_av is within tr_rw
                            self.tr_lb_2[i, j] = tr_lb_2_rw[i, j] #tr_lb_2_av[i, j]
                            self.tr_ub_2[i, j] = tr_ub_2_rw[i, j] #tr_ub_2_av[i, j]
                        else:
                            self.tr_lb_2[i, j] = 0
                            self.tr_ub_2[i, j] = 0
                    
            self.tr_lb_2 = np.clip(self.tr_lb_2, 0.0, 1.0)
            self.tr_ub_2 = np.clip(self.tr_ub_2, 0.0, 1.0)
            
        
        
        
    
   