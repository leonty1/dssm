""" dSSM: A single layer of time-varying SSM in convolutional representations with
time-invariant parameters: lamda and delta 
time-varying parameters: B(t) C(t)
all are real-valued.
"""
import logging
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange, repeat
from scipy import signal
contract = torch.einsum
from omegaconf import DictConfig
import sys


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger
log = get_logger(__name__)

class OptimModule(nn.Module):
    """
    Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters
    Class reused from S4.
    """

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter
        args
            name: name of the buffer/parameter
            tensor: tensor to register
            trainable: whether to register as a trainable parameter (default: False)
            lr: learning rate to use for this parameter (default: None)
            wd: weight decay to use for this parameter (default: None)
        """

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)

def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu'
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.uniform_)
    elif name == 'kaiming_uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'xavier_uniform':
        initializer = torch.nn.init.xavier_uniform_
    elif name == 'normal':
        initializer = torch.nn.init.normal_
    elif name == 'kaiming_normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier_normal':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'trunc_normal':
        initializer = torch.nn.init.trunc_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    elif name == 'half':
        initializer = partial(torch.nn.init.constant_, val=0.5)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer

def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.05, inplace=False)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


_c2r = torch.view_as_real       # complex to real，a+bi --> [a,b]
_r2c = torch.view_as_complex        # real to complex， [a,b] --> a+bi

def reciprocal(x, epsilon=1e-7, clamp=False):
    """ input real or complex number x, returns 1 / x, with bounded norm """
    x_conj = x.conj()
    norm_sq = (x*x_conj).real.clamp(epsilon) if clamp else (x*x_conj + epsilon)
    return x_conj / norm_sq

##############SSM Initialization functions#########
def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix, resued from S5.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P.unsqueeze(1) * P.unsqueeze(0)
    out = torch.tril(A) - torch.diag(torch.arange(N))         
    return -out

def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    hippo = make_HiPPO(N)
    P = torch.sqrt(torch.arange(N) + 0.5)
    B = torch.sqrt(2 * torch.arange(N) + 1.0)
    return hippo, P, B

def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda-[N], low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)
    #
    A = A.type(torch.complex64)
    P = P.type(torch.complex64)
    B = B.type(torch.complex64)

    S = A + P.unsqueeze(1) * P.unsqueeze(0)
    # S = S.type(torch.complex64).cuda()

    S_diag = torch.diagonal(S)  # [N]
    Lambda_real = torch.mean(S_diag) * torch.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = torch.linalg.eig(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B_out = V.conj().T @ B

    return Lambda_real + 1j * Lambda_imag, P, B_out, V, B_orig

def lambda_initializer(channel,head,d_state,Lambda_init,conj_sym,keep_d_state):
    '''
    initialize A
    return them all in real representations
    complex
    Lambda, [Channel, d_state, 2]
    or real
    Lambda, [Channel, N]
    '''
    initializer_list = ['uniform', 'kaiming_uniform', 'xavier_uniform', 'normal', 'kaiming_normal', 'xavier_normal', 'trunc_normal', 'zero', 'one', 'half']

    if Lambda_init == 'hippo':
        if conj_sym and keep_d_state:
            # double the d_state to keep size unchange -- reference S4D in https://github.com/state-spaces/s4
            d_state = d_state*2

        Lambda0, _, B, V, B_orig = make_DPLR_HiPPO(d_state)

        if conj_sym:
            d_state = d_state // 2

        Lambda = Lambda0[:d_state]
        Lambda=Lambda.unsqueeze(0).expand(channel, head, -1).to(torch.cfloat)
        Lambda = _c2r(Lambda)

    elif Lambda_init in initializer_list:
        Lambda_init=get_initializer(Lambda_init)
        Lambda = torch.empty(channel, head, d_state, 2, dtype=torch.float)
        Lambda_init(Lambda)
    else:
        raise ValueError(f"Lambda init {Lambda_init} is not implemented")

    assert Lambda.dim()==4

    Lambda_norm=torch.sqrt(Lambda[...,0]**2+Lambda[...,1]**2)

    return -torch.abs(Lambda_norm)

def hippo_skew_evals(N):
    """ eigenvalues of (Hippo - Hippo.t()) / 2  (largest imag part first) """
    i = torch.arange(N, dtype=torch.float)
    x = 2*i + 1
    Hippo = (x.view(-1,1) * x.view(1,-1)).sqrt().tril(diagonal=-1)  
    Skew = (Hippo - Hippo.t()) / 2                                  
    evals = torch.linalg.eigvals(Skew)                              
    # decreasing order of imag
    return evals[evals.imag.argsort(descending=True)]               

#######################constrain A with negtive valued real part in continuous SSM
def neg_real_Lambda(version, Lambda, max_real_Lambda):
    #['softplus', 'sigmoid', 'Gaussian',  'clip']    - returen real
    if 'softplus' in version:
        Neg_real_lambda =  -F.softplus(Lambda)                     #  [C,S,N] (-∞,0)
    elif 'sigmoid' in version:
        Neg_real_lambda =   -F.sigmoid(Lambda)                     # (-1,0)
    elif 'Gaussian' in version:
        Neg_real_lambda =   -torch.exp(-Lambda**2)                 # (-∞,0)
    elif 'clip' in version:
        Neg_real_lambda =   Lambda.clip(max=-max_real_Lambda)      # (-∞,0)
    else:
        raise NotImplementedError(f"version {version} is not implemented")
    return Neg_real_lambda

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # RMSNorm 计算公式
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight

############################### dSSM layer ##############################
class dSSM(OptimModule):
    '''
    dSSM,  dynamic state space model with time-dependent parameters
    input [B, L, d_model],
    output [B, L, d_model],
    state[B, N, L]
    '''
    def __init__(
            self,
            d_model=1,                  #d_model: the dimension of the input, (B, L, H) 
            d_output=None,              #dimension of the output, default=None, d_in=d_out
            d_s=256,                    #d_state: the dimension of the state, also denoted by N
            bidirectional=False,        #used for noncausal case, like image classification
            channel=1,                  #channels: can be interpreted as a number of ssm in parallel connection, TODO as chunks in future
            head=1,
            dt_min=1e-3,
            dt_max=1e-1,
            d_dt_is_n=False,            #every head has the same scalar dt (False) or not (True)
            trainable=None,
            lr=None,
            Lambda_init='hippo',
            max_real_Lambda=1e-4,
            D_init='one',
            D_value=1,                  #[0, 1, 2]
            l_max=1,                    #the maximum sequence length, also denoted by L
            conj_sym=True,
            keep_d_state=False,         #True N=2N, d_state is the true state size; if false, N denotes half the true state size
            version_A_neg='clip',       # method to ensure negtive real part of Lambda
            max_kernel_length=None,     # max len of SSM kernel to be used, only used for specific setting
            use_bias=False,             #whether linear projection use bias
            sharing_para=True,          #share parameters dt, B, C across heads 
            use_conv=True,              #convolution for U B C, ref. Mamba
            use_conv_act=True,
            use_gate=False,             #using gate mechanism or not, default false, ref mamba-2
            activation='silu',          #internal activation for gate, ref mamba-2, only used when use_gate is True
            use_inner_norm=True,
            **layer_args,
        ):

        super().__init__()

        #input/output size of this block
        self.d_model = d_model          #input size [B,L,d_model]
        if d_output is None:
            self.d_out=d_model          #output size [B,L,d_out], =d_model if not specified.
        else:
            self.d_out=d_output

        #input and output size of SSM
        self.expand=1                   #Not used, expand the ssm size with 2 in ref Mamba, https://arxiv.org/abs/2312.00752
        self.d_ssm_in=(self.expand * self.d_model)
        self.d_ssm_out=(self.expand * self.d_out)
        
        #internal data size of ssm - [channel/C,head/S,...]
        self.channel=channel
        self.head=head                  #multi-head setting of spatial size
        self.h = self.d_ssm_in//head  # d_in_ssm
        self.n = d_s//head              #directly give d_state in each head, not d_s//head
        self.m = self.d_ssm_out // head   
        
        self.bidirectional = bidirectional
        self.dt_min=dt_min
        self.dt_max=dt_max
        self.d_dt_is_n = d_dt_is_n
        self.D_value = D_value
        self.max_real_Lambda=max_real_Lambda
        self.conj_sym=conj_sym
        self.max_kernel_length=max_kernel_length
        self.version_A_neg=version_A_neg
        self.activation = activation
        self.sharing_para=sharing_para

        self.use_conv=use_conv
        self.use_conv_act=use_conv_act
        self.use_gate=use_gate
        self.use_inner_norm=use_inner_norm

        #lernable parameter Lambda, dt, D
        self.Lambda_init=Lambda_init
        Lambda= lambda_initializer(channel=self.channel,head=self.head, d_state=self.n,Lambda_init=Lambda_init, conj_sym=self.conj_sym,keep_d_state=keep_d_state).to('cuda')
        
        #parameterize learnable dt
        if self.d_dt_is_n:
            dt_init = torch.empty(self.channel, self.head, self.n)  # [c,s,n]
        else:
            dt_init = torch.empty(self.channel, self.head)  #[c,s]

        torch.nn.init.uniform_(dt_init, a=0.0, b=1.0)
        log_dt = torch.log(torch.tensor(dt_min)) + dt_init * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))  #[c,s] or [c,s,n]

        self.lr = DictConfig(
            {"log_dt": 1e-3, "Lambda": 1e-3})
        if lr is not None:
            self.lr.update(lr)

        self.trainable = DictConfig({"log_dt": True, "Lambda": True})
        if trainable is not None:
            self.trainable.update(trainable)

        self.register("log_dt", log_dt.cuda(), self.trainable.log_dt, self.lr.log_dt, wd=0.0)
        self.register("Lambda", Lambda.cuda(), self.trainable.Lambda, self.lr.Lambda, wd=0.0)

        # Initlalize D
        D_initializer = get_initializer(D_init)
        if D_value == 0:
            pass
        elif D_value == 1 and self.m == self.h:
            # D is identity matrix, if not trainable, seen as skip connection
            D = torch.ones(self.channel, self.head, self.m)
            if self.trainable.D:
                D_initializer(D)
            self.register("D", D.cuda(), self.trainable.D, self.lr.D, wd=0.0)
        else:
            D = torch.zeros(self.channel, self.head, self.m, self.h, device='cuda')
            D_initializer(D)
            self.register("D", D.cuda(), self.trainable.D, self.lr.D, wd=0.0)
        
        #context-aware parameters U, B, C, delta
        self.U_proj = nn.Linear(self.d_model, self.d_ssm_in, bias=use_bias) #[B, L, self.d_ssm_in]

        if self.sharing_para:
            #all heads have the same B C，sharing across heads
            self.B_dim=self.channel*1*self.h*self.n
            self.B_proj = nn.Linear(self.d_model, self.B_dim, bias=use_bias) #[B, L, CNH]

            self.C_dim=self.channel*1*self.m*self.n
            self.C_proj = nn.Linear(self.d_model, self.C_dim, bias=use_bias) #[B, L, CMN]
            
        else:
            self.B_dim=self.channel*self.head*self.h*self.n
            self.B_proj = nn.Linear(self.d_model, self.B_dim, bias=use_bias) #[B, L, CSNH]

            self.C_dim=self.channel*self.head*self.m*self.n
            self.C_proj = nn.Linear(self.d_model, self.C_dim, bias=use_bias) #[B, L, CSMN]

        #conv1d
        if self.use_conv:
            conv_bias=True
            d_kernel=3
            padding='same'
            self.conv=nn.Conv1d(in_channels=self.d_ssm_in+self.B_dim+self.C_dim, out_channels=self.d_ssm_in+self.B_dim+self.C_dim, kernel_size=d_kernel, padding=padding, groups=self.d_ssm_in+self.B_dim+self.C_dim, bias=conv_bias, padding_mode='zeros')

        if self.use_gate:
            self.gate_linear = nn.Linear(self.d_model, self.d_ssm_out, bias=use_bias)
        
        self.activation_func = Activation(self.activation)

        #inner normalization
        if self.use_inner_norm:
            self.norm = RMSNorm(dim=self.d_ssm_out, eps=1e-5)

        #output linear projection
        self.out_linear = nn.Linear(self.channel*self.d_ssm_out, self.d_out, bias=use_bias)

    def forward(self, u_input, **kwargs):
        """
        ---Convolutional TVSSM for training
        input u: (B L d_model)
        state: (H N) never needed unless you know what you're doing

        Returns: 
        y, output, (B L d_out)
        x, state [b,l,c,s,n,2], real part [b,l,c,s,n,0], image part [b,l,c,s,n,1]
        """
        batch, L, _= u_input.shape

        #LGU linear projection
        if self.use_gate:
            u_gate=self.gate_linear(u_input)
            u_gate=repeat(u_gate, 'b l m -> b l c m', c=self.channel)
            u_gate=self.activation_func(u_gate)
        
        #context-aware U, B, C
        U=self.U_proj(u_input)
        B=self.B_proj(u_input)
        C=self.C_proj(u_input)
        #1dconv
        if self.use_conv:
            ubc=torch.cat((U, B, C), -1)
            ubc=self.conv(ubc.transpose(-2, -1)).transpose(-2, -1)
        
            #gate activation
            if self.use_conv_act:
                ubc=self.activation_func(ubc) #ensure mumeracal stability

            U, B, C = torch.split(ubc, [self.d_ssm_in, self.B_dim, self.C_dim], dim=-1)
        else:
            if self.use_conv_act:
                ubc=torch.cat((U, B, C), -1)
                ubc=self.activation_func(ubc)
                U, B, C = torch.split(ubc, [self.d_ssm_in, self.B_dim, self.C_dim], dim=-1)
        
        U=rearrange(U, 'b l (s h) -> b l s h', s=self.head, h=self.h)   #[b,l,s,h]

        if self.sharing_para:
            B=rearrange(B, 'b l (c n h) -> b l c n h', c=self.channel, n=self.n, h=self.h)
            C=rearrange(C, 'b l (c m n) -> b l c m n', c=self.channel, m=self.m, n=self.n)
        else:
            B=rearrange(B, 'b l (c s n h) -> b l c s n h', c=self.channel, s=self.head, n=self.n, h=self.h)
            C=rearrange(C, 'b l (c s m n) -> b l c s m n', c=self.channel, s=self.head, m=self.m, n=self.n)

        delta = self.log_dt.exp()
        delta = F.softplus(delta)   #ensuring positive delta
        delta=torch.clip(delta, max=self.dt_max, min=self.dt_min)
        
        # L = U.size(1)
        Lk = L if not self.max_kernel_length else min(self.max_kernel_length, L)
        length = torch.arange(Lk).to(self.Lambda)

        #constrain Lambda to ensure negative valued for stability and convergence
        Lambda = neg_real_Lambda(version=self.version_A_neg, Lambda=self.Lambda,max_real_Lambda=self.max_real_Lambda)

        #Discretize the Contimuous SSM
        if self.d_dt_is_n:
            Lambda_dt = contract('csn,csn->csn', Lambda, delta)
        else:
            Lambda_dt = contract('csn,cs->csn', Lambda, delta)

        Identity = torch.ones_like(Lambda_dt)
        B_coef = contract('csn,csn->csn', reciprocal(Lambda), (torch.exp(Lambda_dt)-Identity))

        p=contract('csn,l->lcsn', Lambda_dt, length)

        state_kernel = p.exp()

        #LTV-SSM Mechanism
        if self.sharing_para:
            B_u = contract('blcnh,blsh->blcsn', B, U)
        else:
            B_u = contract('blcsnh,blsh->blcsn', B, U)
        
        #discret B
        B_u=contract('csn,blcsn->blcsn', B_coef, B_u) 

        #reshape for convlution
        B_u=rearrange(B_u, 'b l c s n -> b c s n l')
        state_kernel=rearrange(state_kernel, 'l c s n -> c s n l')

        #bidirectional kernel for non-causal state inference
        if self.bidirectional:
            state_kernel_new = F.pad(state_kernel, (0, Lk)) + F.pad(state_kernel.flip(-1), (Lk, 0))  #bidirectional without additional parameters
        else:
            state_kernel_new = state_kernel


        ############## fast tensor convolution for state inference
        n = L + Lk
        k_f = torch.fft.rfft(state_kernel_new, n=n)  # [c,s,n,2l]
        u_f = torch.fft.rfft(B_u, n=n)  # [b,c,s,n,2l]
        x_f = contract('csnl,bcsnl->bcsnl', k_f, u_f)
        x = torch.fft.irfft(x_f, n=n)[..., :L]
        x=rearrange(x, 'b c s n l -> b l c s n')

        if self.sharing_para:
            Cx = contract('blcmn,blcsn->blcsm', C, x)
        else:
            Cx = contract('blcsmn,blcsn->blcsm', C, x)  

        if self.Lambda_init=='hippo' and self.conj_sym:
            Cx = 2*Cx       #conjugate symmetry

        #output function
        if self.D_value==0:
            # without input
            y = Cx
        elif self.D_value == 1 and self.m == self.h:
            # Compute D term in state space equation - essentially a skip connection
            y = Cx + contract('csm, blsm->blcsm', self.D, U)
        else:
            #affine projection
            y = Cx + contract('csmh, blsh->blcsm', self.D, U)

        y = rearrange(y, 'b l c s m -> b l c (s m)')
        
        if self.use_gate:
            y = y + u_gate
        
        if self.use_inner_norm:
            y=self.norm(y)

        y = rearrange(y, 'b l c m -> b l (c m)')
        y = self.out_linear(y)

        return y, x
    
    @property
    def d_state(self):
        return self.n * self.head

    @property
    def d_output(self):
        return self.d_out