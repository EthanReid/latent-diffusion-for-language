import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
import csv
import timeit
import json
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import wandb
from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from latent_models.t5_latent_model import T5ForConditionalGenerationLatent

import diffusion.constant as constant
import diffusion.optimizer as optimizer
import dataset_utils.text_dataset as text_dataset
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from latent_models.latent_utils import get_latent_model
from evaluation import evaluation
from diffusion.text_denoising_diffusion import GaussianDiffusion
from model.diffusion_transformer import DiffusionTransformer



ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_v'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def l2norm(t):
    return F.normalize(t, dim = -1)

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# normalize variance of noised latent, if scale is not 1

def normalize_z_t_variance(z_t, mask, eps = 1e-5):
    std = rearrange([reduce(z_t[i][:torch.sum(mask[i])], 'l d -> 1 1', partial(torch.std, unbiased = False)) for i in range(z_t.shape[0])], 'b 1 1 -> b 1 1')
    return z_t / std.clamp(min = eps)
    

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def beta_linear_schedule(t, clip_min = 1e-9):
    return torch.exp(-1e-4 - 10 * (t ** 2)).clamp(min = clip_min, max = 1.)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

def n_schedule_value(t, T=1000, start=10, end=100):
    nk = math.ceil(math.sqrt((min(t,T)/T)*((end+1)**2 - start**2)+start**2)-1)+1
    return nk

# converting gamma to alpha, sigma or logsnr

def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha

def alpha_to_shifted_log_snr(alpha, scale = 1):
    return log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2*np.log(scale).item()

def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale = scale)
    return log_snr_to_alpha(shifted_log_snr)

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

#TODO: misspelled boundary
def scaling_for_ddim_boundry(tk, t):
    c_skip = t//tk
    ratio = t/tk
    c_out = torch.where(ratio<1, torch.tensor(1, dtype=tk.dtype, device=tk.device), torch.tensor(0, dtype=tk.dtype, device=tk.device))
    return c_skip, c_out
    
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

class ConsistencyDistillation(nn.Module):
    def __init__(
            self,
            online_model:DiffusionTransformer,
            target_model:DiffusionTransformer,
            diffusion_model:GaussianDiffusion,
            loss_type:str = "l2",
            k=1,
            steps=1,
            both_online=False,
            is_consistency_distillation=False,
            is_mcd = False,
            n_schedule=None
    ):
        super().__init__()
        self.online_model = online_model #init to diffusion weights
        self.target_model = target_model #need to insure that target and online are init with the same weights, this should happen is the caller function
        self.diffusion_model = diffusion_model
        self.loss_type = loss_type
        self.k = k
        self.steps = steps
        self.both_online = both_online
        self.is_consistency_distillation = is_consistency_distillation
        self.forward = self.forward_cd if is_consistency_distillation else self.forward_ct
        self.is_mcd = is_mcd
        if is_mcd:
            if is_consistency_distillation:
                self.forward = self.forward_mcd
            else:
                self.forward = self.forward_mct_z
        self.n_scedule = n_schedule
    
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        elif self.loss_type == 'ground_l2':
            return ConsistencyDistillation.ground_l2
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    
    def ground_l2(online_out, teacher_out, ground_truth, theta=1e-3):
        consistency_loss = F.mse_loss(online_out, teacher_out, reduction="none")
        ground_loss = F.mse_loss(online_out, ground_truth, reduction="none")
        return consistency_loss+(theta*ground_loss)
    
    
    def consistency_model_predictions(self, z_t, mask, t, z_self_cond=None, class_id=None, seq2seq_cond=None, seq2seq_mask=None, sampling=False, cls_free_guidance=1.0, l2_normalize=False, online=True):
        time_to_alpha = self.diffusion_model.sampling_schedule if sampling else self.diffusion_model.train_schedule
        time_cond = time_to_alpha(t)
        #time_cond = right_pad_dims_to(z_t, time_cond)# might not be needed
        model = self.online_model if online else self.target_model
        model_output = model(z_t, mask, time_cond, z_self_cond, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
        if cls_free_guidance!=1.0:
            if exists(class_id):
                unc_class_id = torch.full_like(class_id, fill_value=self.diffusion_model.diffusion_model.num_classes)
            else:
                unc_class_id = None
            unc_model_output = model(z_t, mask, time_cond, z_self_cond, class_id=unc_class_id, seq2seq_cond=None, seq2seq_mask=None)
            model_output = model_output*cls_free_guidance + unc_model_output*(1-cls_free_guidance)
        if l2_normalize:
            assert sampling
            model_output = F.normalize(model_output, dim=-1) * math.sqrt(model_output.shape[-1])
        #pred x_0 with model out:
        c_skip, c_out = scalings_for_boundary_conditions(t)
        c_skip, c_out = [append_dims(x, z_t.ndim) for x in [c_skip, c_out]]
        model_output = c_skip*z_t + c_out*model_output
        return model_output

    def scms(self, shape, lengths, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance=1.0, l2_normalize=False, invert=False, z_t=None, steps=1):
        '''
        Self-Consitional Multi Step sampling
        '''
        assert steps != 0
        batch, device = shape[0], next(self.online_model.parameters()).device

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)
        z_hat_0 = None
        
        times = torch.linspace(1, 0,steps, device=device)
        #times = times.unsqueeze(0)
        #alphas = self.diffusion_model.sampling_schedule(times)
        #alphas = right_pad_dims_to(z_t, alphas)
        
        if self.diffusion_model.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.diffusion_model.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
        #TODO add z_hat_0 = self.consistency_model so that first step is conditioned on prep from start? This is how I do it in train?
        print("scms sample for {} steps".format(str(steps)))
        for i, time in enumerate(times):
            time = time.unsqueeze(0)
            alpha = self.diffusion_model.sampling_schedule(time)
            if i!=0:
                noise = torch.randn_like(z_t)
                z_t = z_hat_0 + math.sqrt(max((alpha)**2 -(1e-9 **2),0))*noise
            z_hat_0 = self.consistency_model_predictions(z_t, mask, time, class_id=class_id, z_self_cond=None, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)   
        return (z_hat_0, mask)

    def mcm_sample(self, shape, lengths, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance=1.0, l2_normalize=False, invert=False, z_t=None, steps=1):
        batch, device = shape[0], next(self.online_model.parameters()).device
        if not exists(z_t):
            z_t = torch.randn(shape, device=device)
        if self.diffusion_model.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.diffusion_model.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)
        print("multi step cm with step = {}".format(steps))
        for i in range(steps):
            t = 1-(i/steps)
            s = t-(1/steps)
            t = torch.tensor(t, device=device).unsqueeze(0)
            s = torch.tensor(s, device=device).unsqueeze(0)
            x_hat = self.consistency_model_predictions(z_t, mask, t=t, class_id=class_id, z_self_cond=None, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)   
            z_t = self.ddim(z_t, x_hat, t, s)
        return (z_t, mask)

    def sample(self, batch_size, length, class_id=None, seq2seq_cond=None, seq2seq_mask=None, cls_free_guidance=1.0, l2_normalize=False, steps=None):
        if steps == None:
            steps = self.steps
        max_seq_len, latent_dim = self.diffusion_model.max_seq_len, self.diffusion_model.latent_dim
        if self.is_mcd:
            return self.mcm_sample((batch_size, max_seq_len, latent_dim), length, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance, l2_normalize, steps=steps)
        return self.scms((batch_size, max_seq_len, latent_dim), length, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance, l2_normalize, steps=steps)
    
    def get_alpha_sigma(self, tk, t, z):
        alpha_tk = self.diffusion_model.sampling_schedule(tk)
        alpha_t = self.diffusion_model.sampling_schedule(t)
        alpha_tk, alpha_t = map(partial(right_pad_dims_to, z), (alpha_tk, alpha_t))
        sigma_tk = (1-alpha_tk)
        sigma_t = (1-alpha_t)
        return alpha_tk.sqrt(), alpha_t.sqrt(), sigma_tk.sqrt(), sigma_t.sqrt()
    def ddim(self, z_t, pred_x, tk, t):
        alpha_tk, alpha_t, sigma_tk, sigma_t = self.get_alpha_sigma(tk, t, z_t)
        c_skip, c_out = scaling_for_ddim_boundry(sigma_tk, sigma_t)
        out = alpha_t*pred_x + ((sigma_t/sigma_tk)*(z_t-alpha_tk*pred_x))
        return c_skip*z_t + c_out*out

    def invDDIM(self, z_t, z_tk, tk, t):
        alpha_tk, alpha_t, sigma_tk, sigma_t = self.get_alpha_sigma(tk, t, z_t)
        x = (z_t-((sigma_t/sigma_tk)*z_tk))/(alpha_t-(alpha_tk*(sigma_t/sigma_tk)))
        return x
    
    def aDDIM(self, z_t, pred_x, ground_x, tk, t, d=None):
        if d==None:
            d = z_t.ndim
        alpha_tk, alpha_t, sigma_tk, sigma_t = self.get_alpha_sigma(tk, t, z_t)
        x_var = (torch.norm(pred_x-ground_x)**2)/d
        eps = (z_t-alpha_tk*pred_x)/sigma_tk
        z_s_var = (alpha_t-alpha_tk*sigma_t/sigma_tk)**2 * x_var
        z_s = alpha_t*pred_x+((sigma_t**2)+(d/torch.norm(eps)**2)*z_s_var)*eps
        return z_s

    def forward_mcd(self, txt_latent, mask, class_id, seq2seq_cond=None, seq2seq_mask=None, return_x_start=False, k=None, train_i=1, *args, **kwargs):
        batch, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.diffusion_model.max_seq_len
        ni = self.n_scedule(train_i)
        T_step = int(np.round(ni/self.steps))
        step = torch.randint(0, self.steps, (batch,), device=device).float()
        n_rel_raw = torch.randint(1, T_step+1, (batch,), device=device).float()
        #n_rel = n_rel_raw/ni
        t_step_raw = step/self.steps
        t_step = t_step_raw/ni
        t_raw = t_step_raw + n_rel_raw/T_step
        t = t_raw/ni
        s_raw = t_raw-1/T_step
        s = s_raw/ni
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, txt_latent)
        
        noise = torch.randn_like(txt_latent)
        z_t = alpha_t*txt_latent + sigma_t*noise

        with torch.no_grad():
            x_teacher = self.diffusion_model.diffusion_model_predictions(z_t=z_t, mask=mask, t=t ,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask).pred_x_start

        z_s = self.aDDIM(z_t, x_teacher, txt_latent, tk=t, t=s, d=d)
        with torch.no_grad():
            x_ref = self.consistency_model_predictions(z_t=z_s, mask=mask, t=s, z_self_cond=None,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, online=self.both_online)
        z_ref = self.ddim(z_s, x_ref, tk=s, t=t_step)

        x = self.consistency_model_predictions(z_t=z_t, mask=mask, t=t, z_self_cond=None,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
        x_inv = self.invDDIM(z_ref, z_t, tk=t, t=t_step)

        loss = self.loss_fn(x,x_inv, reduction="none")
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')

        return loss.mean()

    def forward_mcd_z(self, txt_latent, mask, class_id, seq2seq_cond=None, seq2seq_mask=None, return_x_start=False, k=None, train_i=1, *args, **kwargs):
        batch, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.diffusion_model.max_seq_len
        ni = self.n_scedule(train_i)
        T_step = int(np.round(ni/self.steps))
        step = torch.randint(0, self.steps, (batch,), device=device).float()
        n_rel_raw = torch.randint(1, T_step+1, (batch,), device=device).float()
        #n_rel = n_rel_raw/ni
        t_step_raw = step/self.steps
        t_step = t_step_raw/ni
        t_raw = t_step_raw + n_rel_raw/T_step
        t = t_raw/ni
        s_raw = t_raw-1/T_step
        s = s_raw/ni
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, txt_latent)
        
        noise = torch.randn_like(txt_latent)
        z_t = alpha_t*txt_latent + sigma_t*noise

        with torch.no_grad():
            x_teacher = self.diffusion_model.diffusion_model_predictions(z_t=z_t, mask=mask, t=t ,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask).pred_x_start

        z_s = self.aDDIM(z_t, x_teacher, txt_latent, tk=t, t=s, d=d)
        with torch.no_grad():
            x_ref = self.consistency_model_predictions(z_t=z_s, mask=mask, t=s, z_self_cond=None,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, online=self.both_online)
        z_ref_teacher = self.ddim(z_s, x_ref, tk=s, t=t_step)

        x = self.consistency_model_predictions(z_t=z_t, mask=mask, t=t, z_self_cond=None,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
        z_ref_online = self.ddim(z_t, x, t, t_step)

        loss = self.loss_fn(z_ref_online,z_ref_teacher, reduction="none")
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')
        
        return loss.mean()
    
    def forward_mct_z(self, txt_latent, mask, class_id, seq2seq_cond=None, seq2seq_mask=None, return_x_start=False, k=None, train_i=1, *args, **kwargs):
        batch, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.diffusion_model.max_seq_len
        ni = self.n_scedule(train_i)
        T_step = int(np.round(ni/self.steps))
        step = torch.randint(0, self.steps, (batch,), device=device).float()
        n_rel_raw = torch.randint(1, T_step+1, (batch,), device=device).float()
        #n_rel = n_rel_raw/ni
        t_step_raw = step/self.steps
        t_step = t_step_raw/ni
        t_raw = t_step_raw + n_rel_raw/T_step
        t = t_raw/ni
        s_raw = t_raw-1/T_step
        s = s_raw/ni
        alpha_t, alpha_s, sigma_t, sigma_s = self.get_alpha_sigma(t, s, txt_latent)
        
        noise = torch.randn_like(txt_latent)
        z_t = alpha_t*txt_latent + sigma_t*noise

        x_teacher = txt_latent

        z_s = self.ddim(z_t, x_teacher, tk=t, t=s)
        with torch.no_grad():
            x_ref = self.consistency_model_predictions(z_t=z_s, mask=mask, t=s, z_self_cond=None,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, online=self.both_online)
        z_ref_teacher = self.ddim(z_s, x_ref, tk=s, t=t_step)

        x = self.consistency_model_predictions(z_t=z_t, mask=mask, t=t, z_self_cond=None,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
        z_ref_online = self.ddim(z_t, x, t, t_step)

        loss = self.loss_fn(z_ref_online,z_ref_teacher, reduction="none")
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')
        
        return loss.mean()


    def forward_ct(self, txt_latent, mask, class_id, seq2seq_cond=None, seq2seq_mask=None, return_x_start=False, k=None, *args, **kwargs):
        if k == None:
            k = self.k
        #k = np.random.randint(1,k+1)
        self.target_model.eval()
        self.diffusion_model.eval()
        batch, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.diffusion_model.max_seq_len
        assert l == max_seq_len, f'length must be {max_seq_len}'

        #raw times
        raw_time_n = torch.randint(0,self.diffusion_model.sampling_timesteps-k+1, (batch,), device=device).float() #should this be -k+1, on local it is -k and works, is this the issue?
        raw_time_nk = raw_time_n + k

        #scaled_times
        time_n = raw_time_n/self.diffusion_model.sampling_timesteps
        time_nk = raw_time_nk/self.diffusion_model.sampling_timesteps

        #gaussian noise to be added to latent
        noise = torch.randn_like(txt_latent)

        #alphas, currently only does linear sched
        alpha_n = self.diffusion_model.train_schedule(time_n)
        alpha_n = right_pad_dims_to(txt_latent, alpha_n)
        alpha_nk = self.diffusion_model.train_schedule(time_nk)
        alpha_nk = right_pad_dims_to(txt_latent, alpha_nk)

        z_n = alpha_n.sqrt()*txt_latent + (1-alpha_n).sqrt()*noise
        z_nk = alpha_nk.sqrt()*txt_latent + (1-alpha_nk).sqrt()*noise

        if self.diffusion_model.diffusion_model.class_conditional and self.diffusion_model.diffusion_model.class_unconditional_prob > 0:
            assert exists(class_id)
            class_unconditional_mask = self.diffusion_model.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.diffusion_model.num_classes
        
        z_0_nk = self.consistency_model_predictions(z_t=z_nk, mask=mask, t=time_nk, z_self_cond=None, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
        with torch.no_grad():
            z_0_n = self.consistency_model_predictions(z_t=z_n, mask=mask, t=time_n, z_self_cond=None,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, online=self.both_online)
        #TODO: add if for ground_l2
        #loss = self.loss_fn(z_0_nk,z_0_n, reduction="none")
        loss = self.loss_fn(z_0_nk,z_0_n, txt_latent, theta=5e-1)
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')
        
        return loss.mean()

    def forward_cd(self, txt_latent, mask, class_id, seq2seq_cond=None, seq2seq_mask=None, return_x_start=False, k=None, *args, **kwargs):
        if k == None:
            k = self.k
        k = np.random.randint(1,k+1)
        self.target_model.eval()
        self.diffusion_model.eval()
        batch, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.diffusion_model.max_seq_len
        assert l == max_seq_len, f'length must be {max_seq_len}'

        #raw times
        raw_time_n = torch.randint(0,self.diffusion_model.sampling_timesteps-k+1, (batch,), device=device).float() #should this be -k+1, on local it is -k and works, is this the issue?
        raw_time_nk = raw_time_n + k

        #scaled_times
        time_n = raw_time_n/self.diffusion_model.sampling_timesteps
        time_nk = raw_time_nk/self.diffusion_model.sampling_timesteps

        #gaussian noise to be added to latent
        noise = torch.randn_like(txt_latent)

        #alphas, currently only does linear sched
        #alpha_n = self.diffusion_model.train_schedule(time_n)
        alpha_nk = self.diffusion_model.train_schedule(time_nk)
        alpha_nk = right_pad_dims_to(txt_latent, alpha_nk)

        z_nk = alpha_nk.sqrt()*txt_latent + (1-alpha_nk).sqrt()*noise
        with torch.no_grad():
            #z_psi_n = self.diffusion_model.diffusion_model_predictions(z_t=z_nk, mask=mask, t=time_nk ,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask).pred_x_start
            #z_psi_n = self.diffusion_model.ddpm_predictions(z_t=z_nk, mask=mask, t=time_nk ,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, k=k)
            z_psi_n = self.diffusion_model.ddim_predictions(z_t=z_nk, mask=mask, t=time_nk ,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, k=k)
            #z_psi_n = self.diffusion_model.k_predictions(z_t=z_nk, mask=mask, t=time_nk ,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, k=k)
        #class conditioning
        if self.diffusion_model.diffusion_model.class_conditional and self.diffusion_model.diffusion_model.class_unconditional_prob > 0:
            assert exists(class_id)
            class_unconditional_mask = self.diffusion_model.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.diffusion_model.num_classes
        
        #self conditioning
        z_hat_0_nk = None
        z_hat_0_n = None
        """
        if self.diffusion_model.self_condition and (random.random() < self.diffusion_model.train_prob_self_cond):
            with torch.no_grad():
                z_hat_0_nk = self.consistency_model_predictions(z_t=z_nk, mask=mask, t=time_nk, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                z_hat_0_n = self.consistency_model_predictions(z_t=z_psi_n, mask=mask, t=time_n, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                if self.diffusion_model.l2_normalize:
                    z_hat_0_nk = F.normalize(z_hat_0_nk, dim=-1) * math.sqrt(z_hat_0_nk.shape[-1])
                    z_hat_0_n = F.normalize(z_hat_0_n, dim=-1) * math.sqrt(z_hat_0_n.shape[-1])
        """
        z_0_nk = self.consistency_model_predictions(z_t=z_nk, mask=mask, t=time_nk, z_self_cond=z_hat_0_nk, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
        with torch.no_grad():
            z_0_n = self.consistency_model_predictions(z_t=z_psi_n, mask=mask, t=time_n, z_self_cond=z_hat_0_n,class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, online=self.both_online)

        loss = self.loss_fn(z_0_nk,z_0_n, reduction="none")
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')
        
        return loss.mean()

class Trainer(object):
    def __init__(
        self,
        args,
        consistency:ConsistencyDistillation,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        save_and_sample_every = 5000,
        save_every = 1000,
        num_samples = 25,
        seq2seq_candidates = 10,
        seq2seq_train_context_encoder = False,
        results_folder = './results',
        amp = False,
        mixed_precision = 'no',
        decoding_loss = False,
        decoding_loss_weight = 1.0,
        init_models = False,
        default_models = False
    ):
        super().__init__()

        self.seed = 42
        set_seeds(self.seed)

        self.args = args

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=90))

        self.accelerator = Accelerator(
            mixed_precision = mixed_precision,
            log_with='wandb',
            kwargs_handlers=[ddp_kwargs, init_process_kwargs]
        )
        self.num_devices = self.accelerator.num_processes
        args.num_devices = self.num_devices

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = file_utils.get_output_dir(args)
                with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                    json.dump(args.__dict__, f, indent=2)
            results_folder = args.output_dir
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        

        self.consistency = consistency
        self.decoding_loss = decoding_loss
        self.decoding_loss_weight = decoding_loss_weight

        self.num_samples = num_samples
        self.seq2seq_candidates = seq2seq_candidates
        self.save_and_sample_every = save_and_sample_every
        self.save_every = save_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.max_seq_len = self.consistency.diffusion_model.max_seq_len

        self.latent_model_path = args.latent_model_path

        self.enc_dec_model = args.enc_dec_model

        self.k = args.k
        self.steps = args.steps

        # Init Encoder-decoder model
        if 'bart' in args.enc_dec_model:
            self.bart_model = BartForConditionalGeneration.from_pretrained(args.enc_dec_model)
        elif 'flan-t5' in args.enc_dec_model:
            self.bart_model = T5ForConditionalGeneration.from_pretrained(args.enc_dec_model, torch_dtype=torch.bfloat16)
        elif 'mt5' in args.enc_dec_model:
            self.bart_model = MT5ForConditionalGeneration.from_pretrained(args.enc_dec_model, torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f'invalid enc_dec_model {args.enc_dec_model}')
        self.tokenizer = AutoTokenizer.from_pretrained(args.enc_dec_model)

        self.consistency.diffusion_model.using_latent_model = False
        self.seq2seq = self.consistency.diffusion_model.diffusion_model.seq2seq
        self.class_conditional = self.consistency.diffusion_model.diffusion_model.class_conditional
        self.seq2seq_unconditional_prob = self.consistency.diffusion_model.seq2seq_unconditional_prob
        self.best_seq2seq_metric = 0
        self.context_tokenizer = None
        self.ema_decay = ema_decay
        if args.latent_model_path:
            device = self.accelerator.device
            with open(os.path.join(args.latent_model_path, 'args.json'), 'rt') as f:
                latent_model_args = json.load(f)
            
            latent_argparse = argparse.Namespace(**latent_model_args)
            self.consistency.diffusion_model.context_encoder = self.bart_model.get_encoder()
            self.seq2seq_train_context_encoder = seq2seq_train_context_encoder
            if seq2seq_train_context_encoder:
                for param in self.consistency.diffusion_model.context_encoder.parameters():
                    param.requires_grad = True
            else:
                for param in self.consistency.diffusion_model.context_encoder.parameters():
                    param.requires_grad = False

            self.context_tokenizer = self.tokenizer
            self.bart_model, self.tokenizer, _ = get_latent_model(latent_argparse)
            data = torch.load(os.path.join(args.latent_model_path, 'model.pt'), map_location=device)
            self.bart_model.load_state_dict(data['model'])
            diffusion_data = torch.load(os.path.join(args.diffusion_model_path, 'model.pt'), map_location=device)
            self.consistency.diffusion_model.load_state_dict(diffusion_data['model'])
            if init_models:
                for online_param, target_param, diffusion_param in zip(self.consistency.online_model.parameters(), self.consistency.target_model.parameters(), self.consistency.diffusion_model.parameters()):
                    online_param.data.copy_(diffusion_param.clone().detach())
                    target_param.data.copy_(diffusion_param.clone().detach())
            elif default_models:
                for online_param, target_param in zip(self.consistency.online_model.parameters(), self.consistency.target_model.parameters()):
                    target_param.data.copy_(online_param.clone().detach())
            self.consistency.diffusion_model.max_seq_len = self.bart_model.num_encoder_latents
            self.num_encoder_latents = self.bart_model.num_encoder_latents
            self.consistency.diffusion_model.using_latent_model = True
            self.consistency.diffusion_model.l2_normalize = (hasattr(self.bart_model, 'l2_normalize_latents') and self.bart_model.l2_normalize_latents)
            if self.consistency.diffusion_model.l2_normalize:
                assert not args.normalize_latent
            for param in self.bart_model.parameters():
                param.requires_grad = False
        self.using_latent_model = self.consistency.diffusion_model.using_latent_model
        self.bart_model.eval()
            

        # dataset and dataloader
        self.dataset_name = dataset_name
        dataset = text_dataset.get_dataset(dataset_name,)

        self.dataset = dataset.shuffle(seed=self.seed)
        if args.eval_test:
            self.num_samples = min(self.num_samples,len(self.dataset['test']))
            print(f'Using {self.num_samples} samples for evaluation')
        else:
            self.num_samples = min(self.num_samples,len(self.dataset['valid']))
            print(f'Using {self.num_samples} samples for evaluation')
        # Subsample train and val splits for computing language generation during runtime
        
        self.train_val_dataloader = text_dataset.get_dataloader(args, dataset['train'].select(range(1000)), self.bart_model.config, self.tokenizer, self.max_seq_len, shuffle=False, context_tokenizer=self.context_tokenizer)
        if args.resume_training:
            dataset['train'] = dataset['train'].shuffle()
        self.dataloader = text_dataset.get_dataloader(args, self.dataset['train'], self.bart_model.config, self.tokenizer, self.max_seq_len, context_tokenizer=self.context_tokenizer)
        self.val_dataloader = text_dataset.get_dataloader(args, self.dataset['valid'], self.bart_model.config, self.tokenizer, self.max_seq_len, shuffle=False, context_tokenizer=self.context_tokenizer)
        self.test_dataloader = text_dataset.get_dataloader(args, self.dataset['test'], self.bart_model.config, self.tokenizer, self.max_seq_len, shuffle=False, context_tokenizer=self.context_tokenizer)

        if not self.seq2seq:
            training_lengths = [min(sum(self.dataloader.dataset[idx]['attention_mask']), self.max_seq_len) for idx in range(self.dataloader.dataset.num_rows)]
            length_counts = Counter(training_lengths)
            probs = torch.tensor([length_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.max_seq_len+1)])
            assert probs[0] == 0, 'Can\'t have examples of length 0'
            self.length_categorical = torch.distributions.Categorical(probs=probs)

        if self.class_conditional:
            training_labels = [self.dataloader.dataset[idx]['label'] for idx in range(self.dataloader.dataset.num_rows)]
            label_counts = Counter(training_labels)
            probs = torch.tensor([label_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.diffusion.diffusion_model.num_classes)])
            self.class_categorical = torch.distributions.Categorical(probs=probs)
        
        # optimizer

        self.opt = optimizer.get_adamw_optimizer(self.consistency.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay) #is self.consistency for params, should it be online model?

        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps*self.num_devices,
            num_training_steps=train_num_steps*self.num_devices,
        )

        # for logging results in a folder periodically
        
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.consistency, self.bart_model, self.opt, self.dataloader, self.lr_scheduler = self.accelerator.prepare(self.consistency, self.bart_model, self.opt, self.dataloader, lr_scheduler)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    #TODO
    def save(self, best=False):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'online_model': self.accelerator.get_state_dict(self.consistency.online_model),
            'target_model': self.accelerator.get_state_dict(self.consistency.target_model),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.lr_scheduler.state_dict(),
        }
        if best:
            torch.save(data, str(self.results_folder / f'best_model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model.pt'))

    #TODO
    def load(self, file_path=None, best=False, init_only=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        if best:
            data = torch.load(str(file_path / f'best_model.pt'), map_location=device)
        else:
            data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.consistency)
        # For backwards compatibility with earlier models
        model.online_model.load_state_dict(data['online_model'])
        model.target_model.load_state_dict(data['target_model'])
        self.opt.load_state_dict(data['opt'])
        if init_only:
            return
        self.step = data['step']
        
        if 'scheduler' in data:
            self.lr_scheduler.load_state_dict(data['scheduler'])
        # For backwards compatibility with earlier models
        
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    #TODO
    def log_reference_metrics(self, test=False):
        accelerator = self.accelerator
        if test:
            train_subset = self.dataset['train']['text'][:self.num_samples]
            train_subset2 = self.dataset['train']['text'][self.num_samples:(2*self.num_samples)] 
            test_subset = self.dataset['test']['text'][:self.num_samples]
            self.reference_dict['reference/test_perplexity'] = evaluation.compute_perplexity(test_subset, device=self.args.device)
            for mauve_model_id in ["gpt2-large"]:
                self.reference_dict[f'reference/{mauve_model_id}_train_test_mauve'], _ = evaluation.compute_mauve(train_subset, test_subset, mauve_model_id)
                self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(train_subset, train_subset2, mauve_model_id)
                ngram_metrics = evaluation.compute_diversity(test_subset)
            for k, v in ngram_metrics.items():
                self.reference_dict[f"reference/test_{k}"] = v
            self.reference_dict[f"reference/test_memorization"] = evaluation.compute_memorization(test_subset, self.dataset['train']['text'])
            self.reference_dict['reference/test_unique_wordcount'] = evaluation.compute_wordcount(test_subset)
            return

        val_subset = self.dataset['valid']['text'][:self.num_samples]
        train_subset = self.dataset['train']['text'][:self.num_samples]
        train_subset2 = self.dataset['train']['text'][self.num_samples:(2*self.num_samples)] 
        self.reference_dict['reference/train_perplexity'] = evaluation.compute_perplexity(train_subset, device=self.args.device)
        self.reference_dict['reference/val_perplexity'] = evaluation.compute_perplexity(val_subset, device=self.args.device)
        for mauve_model_id in ["gpt2-large"]:
            self.reference_dict[f'reference/{mauve_model_id}_train_val_mauve'], _ = evaluation.compute_mauve(train_subset, val_subset, mauve_model_id)
            self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(train_subset, train_subset2, mauve_model_id)
        ngram_metrics = evaluation.compute_diversity(val_subset)
        for k, v in ngram_metrics.items():
            self.reference_dict[f"reference/val_{k}"] = v
        ngram_metrics = evaluation.compute_diversity(train_subset)
        for k, v in ngram_metrics.items():
            self.reference_dict[f"reference/train_{k}"] = v
        self.reference_dict[f"reference/val_memorization"] = evaluation.compute_memorization(val_subset, self.dataset['train']['text'])
        self.reference_dict['reference/train_unique_wordcount'] = evaluation.compute_wordcount(train_subset)
        self.reference_dict['reference/val_unique_wordcounts'] = evaluation.compute_wordcount(val_subset)
        if self.accelerator.device == "cuda":
            torch.cuda.empty_cache() 
            
    #TODO   
    @torch.no_grad()
    def sample(self, num_samples=None, class_id=None, seed=42, test=False, cls_free_guidance=1.0, steps=None):
        num_samples = default(num_samples, self.num_samples)
        accelerator = self.accelerator
        device = accelerator.device
        if self.accelerator.device == "cuda":
            torch.cuda.empty_cache() 
        if steps == None:
            steps = self.args.steps

        self.consistency.eval()

        # Extract references
        reference_texts = {}
        if exists(class_id):
            for filter_class_id in range(self.consistency.diffusion_model.diffusion_model.num_classes):
                filtered_dataset = self.dataset.filter(lambda example: example["label"]==filter_class_id)
                if test:
                    reference_texts[f'ref{filter_class_id}_test'] = filtered_dataset['test']['text']
                    continue
                reference_texts[f'ref{filter_class_id}_val'] = filtered_dataset['valid']['text']
                reference_texts[f'ref{filter_class_id}_train'] = filtered_dataset['train']['text']
            
            for key, reference_text in reference_texts.items():
                num_samples = min(num_samples, len(reference_text))
            reference_texts = {k: v[:num_samples] for k, v in reference_texts.items()}
        else:
            if test:
                reference_texts[f'test'] = self.dataset['test']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]
            else:
                reference_texts['val'] = self.dataset['valid']['text'][:num_samples]
                reference_texts['train'] = self.dataset['train']['text'][:num_samples]

        milestone = self.step // self.save_and_sample_every
        # Stores generation outputs for each strategy
        all_texts_lists = {k:[] for k,_ in constant.generate_kwargs.items()}    

        torch.manual_seed(seed)
        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.class_conditional:
                if self.consistency.diffusion_model.diffusion_model.class_unconditional_prob > 0:
                    return torch.tensor([self.consistency.diffusion_model.diffusion_model.num_classes]*n, dtype=torch.long, device=device)
                return self.class_categorical.sample((n,)).to(device)
            return None
        # Loop until enough senetences have been generated across all strategies 
        while min([len(all_texts_lists[ele]) for ele in all_texts_lists]) < num_samples:
            batches = num_to_groups(num_samples-min([len(all_texts_lists[ele]) for ele in all_texts_lists]), max(self.eval_batch_size,self.train_batch_size))
            #consistency sample is not implemented!
            model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.consistency.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=get_class_id(n), cls_free_guidance=cls_free_guidance, steps=steps)), batches))
            
            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)
                if self.args.normalize_latent:
                    latents = self.consistency.diffusion_model.unnormalize_latent(latents)
                for k, kwargs in constant.generate_kwargs.items():
                    if self.latent_model_path:
                        attention_mask = None
                        encoder_output = BaseModelOutput(last_hidden_state=self.bart_model.get_decoder_input(latents.clone())) #this is the "reconstructor"
                    else:
                        attention_mask = mask.clone()
                        encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=attention_mask, **kwargs) #decoding step
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                    all_texts_lists[k].extend(texts_list)
        
        assert min([len(all_texts_lists[ele]) for ele in all_texts_lists]) >= num_samples
        text_generations = {k:v[:num_samples] for k,v in all_texts_lists.items()} 

        metrics = {}

        self.consistency.to('cpu')
        if self.accelerator.device == "cuda":
            torch.cuda.empty_cache() 
        for strategy, all_texts_list in text_generations.items():
            class_id_prefix = f'cond{class_id}_' if exists(class_id) else ''
            file_utils.save_text_samples(all_texts_list, os.path.join(self.results_folder, f'{"eval-" if self.args.eval else ""}{f"eval{seed}-" if self.args.eval_test else ""}{class_id_prefix}{strategy}-sample-{milestone}.txt'))
            metrics[f"model/{strategy}/{class_id_prefix}perplexity"] = evaluation.compute_perplexity(all_texts_list)
            metrics[f"model/{strategy}/{class_id_prefix}unique_wordcount"] = evaluation.compute_wordcount(all_texts_list)
            ngram_metrics = evaluation.compute_diversity(all_texts_list)
            for k, v in ngram_metrics.items():
                metrics[f"model/{strategy}/{class_id_prefix}{k}"] = v
            metrics[f"model/{strategy}/{class_id_prefix}memorization"] = evaluation.compute_memorization(all_texts_list, self.dataset['train']['text'])
            table = wandb.Table( 
                columns=['Samples'], data=[[text] for text in all_texts_list])
            accelerator.log({f"model/{strategy}/{class_id_prefix}samples": table}, self.step)

            # Only evaluate MAUVE if generations are reasonable to speed up validation early on
            if metrics[f"model/{strategy}/{class_id_prefix}perplexity"] > 5000:
                continue

            for mauve_model_id in ["gpt2-large"]:
                for key, reference_text in reference_texts.items():
                    metrics[f"model/{strategy}/{mauve_model_id}_{class_id_prefix}{key}_mauve"], _ = evaluation.compute_mauve(all_texts_list, reference_text, mauve_model_id)

        if len(self.reference_dict) == 0 or test:
            self.log_reference_metrics(test)
        if test:
            metrics_dict = {**metrics,**self.reference_dict}
            metrics_dict = {f'{k}_seed{seed}':v for k,v in metrics_dict.items()}
            accelerator.log(metrics_dict, self.step)
            print(metrics_dict)
        else:
            accelerator.log({**metrics,**self.reference_dict}, self.step)
        if self.accelerator.device == "cuda":
            torch.cuda.empty_cache() 
        self.consistency.to(device)

    #TODO
    @torch.no_grad()
    def sample_seq2seq(self, num_samples=None, split='val', seed=42, num_candidates=None, cls_free_guidance=1.0,):
        raise NotImplementedError
        assert split in ['train', 'val', 'test']
        num_samples = default(num_samples, self.num_samples) if split != 'test' else len(self.dataset['test'])
        num_candidates = default(num_candidates, self.seq2seq_candidates)
        accelerator = self.accelerator
        device = accelerator.device

        self.ema.ema_model.eval()

        # Extract references
        reference_texts = []
        source_texts = []
        pred_texts = []

        torch.manual_seed(seed)

        if split == 'val':
            dataloader = self.val_dataloader
            prefix = ''
        elif split == 'train':
            dataloader = self.train_val_dataloader
            prefix = 'train/'
        elif split == 'test':
            dataloader = self.test_dataloader
            prefix = 'test/'
        else:
            raise ValueError(f'invalid split {split}')
        
        diffusion = accelerator.unwrap_model(self.diffusion)
        prefix += f'guide{cls_free_guidance}/' if cls_free_guidance != 1.0 else ''
        for batch in dataloader:
            data = batch.to(device)
            seq2seq_cond = diffusion.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
            seq2seq_mask = data['cond_attention_mask'].bool()
            pred_cand_list = []
            ref_cand_list = []
            source_cand_list = []
            gen_kwargs = constant.generate_kwargs['beam']
            gen_kwargs['max_length'] = self.args.max_seq_len
            for _ in range(num_candidates):
                l2_normalize = (hasattr(self.bart_model, 'l2_normalize_latents') and self.bart_model.l2_normalize_latents)
                latents, mask = self.ema.ema_model.sample(batch_size=seq2seq_cond.shape[0], length=None, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
                if self.args.normalize_latent:
                    latents = self.ema.ema_model.unnormalize_latent(latents)
                if self.latent_model_path:
                    attention_mask = None
                    encoder_output = BaseModelOutput(last_hidden_state=self.bart_model.get_decoder_input(latents.clone()))
                else:
                    attention_mask = mask.clone()
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone())
                sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=attention_mask, **gen_kwargs)
                texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in sample_ids]
                pred_cand_list.append(texts_list)

                ref_cand_list.append([self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in data['input_ids']])
                source_cand_list.append([self.context_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in data['cond_input_ids']])
            assert len(pred_cand_list) == num_candidates
            assert len(ref_cand_list) == num_candidates
            assert len(source_cand_list) == num_candidates
            pred_texts.extend([val for tup in zip(*pred_cand_list) for val in tup])
            reference_texts.extend([val for tup in zip(*ref_cand_list) for val in tup])
            source_texts.extend([val for tup in zip(*source_cand_list) for val in tup])
            if len(pred_texts) >= num_samples*num_candidates:
                break
        assert len(pred_texts) == len(reference_texts) == len(source_texts)
        assert len(pred_texts) >= num_samples*num_candidates
        pred_texts = pred_texts[:num_samples*num_candidates]
        reference_texts = reference_texts[:num_samples*num_candidates]
        source_texts = source_texts[:num_samples*num_candidates]

         # Save samples and references to json
        if split == 'test':
            samples_dict = {'pred_texts': pred_texts, 'reference_texts': reference_texts, 'source_texts': source_texts}
            save_path = os.path.join(self.results_folder, f'{prefix}_seq2seq_{split}_samples.json')    
            # Create dir if it doesn't exist   
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            with open(os.path.join(save_path), 'w') as f:
                json.dump(samples_dict, f)

        # Log samples
        # source | reference | pred
        columns = ['source', 'reference', 'pred']
        data = []
        for i in range(len(reference_texts)):
            row = [source_texts[i], reference_texts[i], pred_texts[i]]
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        accelerator.log({f"seq2seq/{prefix}{split}_samples": table}, self.step)

        # Compute metrics
        metrics = {}

        if 'wmt' in self.dataset_name:
            tokenize = 'intl' if self.dataset_name == 'wmt14-en-de' else '13a'

            if num_candidates > 1:
                mbr_sacrebleu_scores = np.zeros((num_samples, num_candidates))
                for i in range(num_candidates):
                    pred_texts_i = pred_texts[i::num_candidates]
                    for j in range(num_candidates):
                        if j == i:
                            continue
                        ref_texts_j = pred_texts[j::num_candidates]
                        sacrebleu_arr = np.array([evaluation.compute_sacrebleu([pred], [ref], tokenize=tokenize, use_effective_order=True) for pred, ref in zip(pred_texts_i, ref_texts_j)])
                        mbr_sacrebleu_scores[:, i] += sacrebleu_arr
                best_indices = np.argmax(mbr_sacrebleu_scores, axis=1)
                best_predictions = [pred_texts[i*num_candidates + idx] for i, idx in enumerate(best_indices)]
                if split == 'test':
                    gt_reference_texts = self.dataset['test']['text'][:num_samples]
                elif split == 'val':
                    gt_reference_texts = self.dataset['valid']['text'][:num_samples]
                elif split == 'train':
                    gt_reference_texts = reference_texts[::num_candidates]
                else:
                    raise NotImplementedError
                metrics[f'model/seq2seq/{prefix}mbr_sacrebleu'] = evaluation.compute_sacrebleu(best_predictions, gt_reference_texts, tokenize=tokenize)
        else:
            # Get oracle rouge
            raw_rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts, use_aggregator=False)
            # Compute the max rouge score across num_candidates
            for k, v in raw_rouge_metrics.items():
                np_metric = np.array(v).reshape(num_samples, num_candidates)
                np_metric = np.max(np_metric, axis=1)
                metrics[f"model/seq2seq/{prefix}oracle_{k}"] = np_metric.mean().item()

            if num_candidates > 1:
                mbr_rouge_scores = np.zeros((num_samples, num_candidates))
                for i in range(num_candidates):
                    pred_texts_i = pred_texts[i::num_candidates]
                    for j in range(num_candidates):
                        if j == i:
                            continue
                        ref_texts_j = pred_texts[j::num_candidates]
                        rouge2_arr = np.array(evaluation.compute_rouge(pred_texts_i, ref_texts_j, use_aggregator=False)['rouge2'])
                        mbr_rouge_scores[:, i] += rouge2_arr
                best_indices = np.argmax(mbr_rouge_scores, axis=1)
                best_predictions = [pred_texts[i*num_candidates + idx] for i, idx in enumerate(best_indices)]
                mbr_rouge_metrics = evaluation.compute_rouge(best_predictions, reference_texts[::num_candidates])
                for k, v in mbr_rouge_metrics.items():
                    metrics[f"model/seq2seq/{prefix}mbr_{k}"] = v
                metrics[f'model/seq2seq/{prefix}mbr_bertscore'] = evaluation.compute_bertscore(best_predictions, reference_texts[::num_candidates])

        # Get every num_candidates samples
        pred_texts = pred_texts[::num_candidates]
        reference_texts = reference_texts[::num_candidates]
        source_texts = source_texts[::num_candidates]
        
        if 'wmt' in self.dataset_name:
            save_path = os.path.join(self.results_folder, f'{prefix}{split}_samples.txt')   
            # Create dir if it doesn't exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            file_utils.save_text_samples(pred_texts, save_path)
            tokenize = 'intl' if self.dataset_name == 'wmt14-en-de' else '13a'
            # Compute BLEU
            if split == 'test':
                assert num_samples == len(self.dataset['test']['text'])
                reference_texts = self.dataset['test']['text'][:num_samples]
            elif split == 'val':
                reference_texts = self.dataset['valid']['text'][:num_samples]
            assert len(pred_texts) == len(reference_texts)
            sacrebleu_score = evaluation.compute_sacrebleu(pred_texts, reference_texts, tokenize=tokenize)
            metrics[f"model/seq2seq/{prefix}sacrebleu"] = sacrebleu_score
            if metrics[f'model/seq2seq/{prefix}sacrebleu'] > self.best_seq2seq_metric and split == 'val' and cls_free_guidance == 1.0:
                self.best_seq2seq_metric = metrics[f'model/seq2seq/{prefix}sacrebleu']
                self.save(best=True)
        else:
            rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts)
            for k, v in rouge_metrics.items():
                metrics[f"model/seq2seq/{prefix}{k}"] = v

            if rouge_metrics['rougeL'] > self.best_seq2seq_metric and split == 'val':
                self.best_seq2seq_metric = rouge_metrics['rougeL']
                self.save(best=True)

            rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts, use_stemmer=True)
            for k, v in rouge_metrics.items():
                metrics[f"model/seq2seq/{prefix}stem_{k}"] = v

            shuffled_pred_texts = random.sample(pred_texts, len(pred_texts))
            shuffled_rouge_metrics = evaluation.compute_rouge(shuffled_pred_texts, reference_texts)
            for k, v in shuffled_rouge_metrics.items():
                metrics[f"model/seq2seq/{prefix}shuffled_{k}"] = v

            metrics[f"model/seq2seq/{prefix}perplexity"] = evaluation.compute_perplexity(pred_texts)
            metrics[f"model/seq2seq/{prefix}unique_wordcount"] = evaluation.compute_wordcount(pred_texts)
            ngram_metrics = evaluation.compute_diversity(pred_texts)
            for k, v in ngram_metrics.items():
                metrics[f"model/seq2seq/{prefix}{k}"] = v
            metrics[f"model/seq2seq/{prefix}memorization"] = evaluation.compute_memorization(pred_texts, self.dataset['train']['text'])
            metrics[f"model/seq2seq/{prefix}bertscore"] = evaluation.compute_bertscore(pred_texts, reference_texts)
        
        accelerator.log(metrics, self.step)
        print(metrics)
        if self.accelerator.device == "cuda":
            torch.cuda.empty_cache() 

    @torch.no_grad()
    def update_ema(self, online_params, target_params, rate):
        for online, target in zip(online_params, target_params):
            target.detach().mul(rate).add(online, alpha=1-rate)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                #TODO center and normalize BART latent space with empirical est. of mean/var.

                total_loss = 0.
                decoding_loss = 0.
                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter).to(device)
                    with torch.no_grad():
                        encoder_outputs = self.bart_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                        if self.using_latent_model:
                            latent = self.bart_model.get_diffusion_latent(encoder_outputs, data['attention_mask'])      
                        else:                      
                            latent = encoder_outputs.last_hidden_state
                        
                        if self.args.normalize_latent:
                            if self.step==0 and grad_accum_step==0:
                                if self.using_latent_model:
                                    latent_vecs = rearrange(latent, 'b s d -> (b s) d')
                                else:
                                    latent_vecs = torch.cat([latent[i][:torch.sum(data['attention_mask'][i])] for i in range(latent.shape[0])], dim=0)
                                
                                # Add mean stats to model and EMA wrapper
                                self.consistency.diffusion_model.latent_mean = torch.mean(latent_vecs, dim=0)
                                #self.ema.ema_model.latent_mean = self.diffusion.latent_mean

                                # Add var stats to model and EMA wrapper
                                self.consistency.diffusion_model.latent_scale = torch.std(latent_vecs-self.diffusion.latent_mean, unbiased=False)

                                #self.ema.ema_model.latent_scale = self.diffusion.latent_scale
                            latent = self.consistency.diffusion_model.normalize_latent(latent)
                        
                    seq2seq_cond = None
                    seq2seq_mask = None
                    with accelerator.autocast():
                        if self.seq2seq and random.random() < (1-self.seq2seq_unconditional_prob):
                            if self.num_devices > 1:
                                seq2seq_cond = self.consistency.diffusion_model.module.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
                            else:
                                seq2seq_cond = self.consistency.diffusion_model.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
                            seq2seq_mask = data['cond_attention_mask'].bool()

                    if self.using_latent_model:
                        mask = torch.ones(latent.shape[0], self.num_encoder_latents, dtype=torch.bool).to(device)
                    else:
                        mask = data['attention_mask'].bool()
                    if self.decoding_loss:
                        raise NotImplementedError
                    else:
                        if self.args.is_mcd:
                            loss = self.consistency(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, train_i=self.step)
                        else:
                            loss = self.consistency(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)                

                accelerator.clip_grad_norm_(self.consistency.online_model.parameters(), self.args.clip_grad_norm)
                grad_norm = compute_grad_norm(self.consistency.online_model.parameters())
                accelerator.wait_for_everyone()
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()
                self.update_ema(self.consistency.online_model.parameters(), self.consistency.target_model.parameters(), self.ema_decay)
                accelerator.wait_for_everyone()

                self.step += 1
                if self.step != 0 and self.step % self.save_every == 0:
                    self.save()
                if accelerator.is_main_process:
                    logs = {
                        "loss": total_loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "step": self.step, 
                        "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), 
                        "samples": self.step*self.train_batch_size*self.gradient_accumulate_every*self.num_devices
                    }
                    if self.decoding_loss:
                        logs['decoding_loss'] = decoding_loss

                    # Log to WandB
                    if self.step % 50 == 0:
                        self.consistency.eval()
                        #self.ema.ema_model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            total_val_ema_loss = 0.
                            for grad_accum_step in range(self.gradient_accumulate_every):
                                data = next(self.val_iter).to(device)
                                
                                encoder_outputs = self.bart_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                                if self.using_latent_model:
                                    latent = self.bart_model.get_diffusion_latent(encoder_outputs, data['attention_mask'])      
                                else:                      
                                    latent = encoder_outputs.last_hidden_state
                                
                                if self.args.normalize_latent:
                                    latent = self.consistency.diffusion_model.normalize_latent(latent)
                                
                                seq2seq_cond = None
                                seq2seq_mask = None
                                if self.seq2seq and random.random() < (1-self.seq2seq_unconditional_prob):
                                    with torch.no_grad():
                                        if self.num_devices > 1:
                                            seq2seq_cond = self.consistency.diffusion_model.module.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
                                        else:
                                            seq2seq_cond = self.consistency.diffusion_model.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
                                    seq2seq_mask = data['cond_attention_mask'].bool()
                                
                                if self.using_latent_model:
                                    mask = torch.ones((latent.shape[0], self.num_encoder_latents), dtype=torch.bool).to(device)
                                else:
                                    mask = data['attention_mask'].bool()
                                if self.args.is_mcd:
                                    loss = self.consistency(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, train_i = self.step)
                                else:
                                    loss = self.consistency(latent, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                                loss = loss / self.gradient_accumulate_every
                                total_val_loss += loss.item()
                            logs["val_loss"] = total_val_loss 
                            pbar.set_postfix(**logs)  
                        self.consistency.train()
                    accelerator.log(logs, step=self.step)              
                    if self.step % self.save_and_sample_every == 0:
                        if self.seq2seq:
                            if 'wmt' in self.args.dataset_name:
                                for guidance_strength in [1.0, 2.0]:
                                    self.sample_seq2seq(cls_free_guidance=guidance_strength, incremental=False)
                            else:
                                self.sample_seq2seq()
                            self.sample_seq2seq(split='train')
                        else:
                            self.sample()
                        if self.class_conditional:
                            for class_id in range(self.consistency.diffusion_model.diffusion_model.num_classes):
                                self.sample(num_samples=100, class_id=class_id)
                        self.save()
                        
                        self.consistency.train() 
                pbar.update(1)
            accelerator.wait_for_everyone()
        self.save()
        accelerator.print('training complete')