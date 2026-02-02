import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean, MultitaskMean, ConstantMean
from gpytorch.kernels import MaternKernel, MultitaskKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.model import Model
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError


def fit_gpytorch_mll_with_adam_fallback( # added for optimization failure (from BO4Mob) 
    mll: ExactMarginalLogLikelihood,
    max_adam_iters: int = 200,
    lr: float = 0.1,
    verbose: bool = True,
) -> bool:
    """
    Fit MLL with fit_gpytorch_mll (L-BFGS-B); on failure, fall back to Adam.
    Returns True if fit succeeded (either method).
    """
    try:
        fit_gpytorch_mll(mll)
        if verbose:
            print("GP fit successful (L-BFGS-B)")
        return True
    except Exception as e:
        if verbose:
            print(f"L-BFGS-B fit failed ({e}), falling back to Adam ({max_adam_iters} steps)")
        mll.train()
        optimizer = torch.optim.Adam(mll.parameters(), lr=lr)
        model = mll.model
        for i in range(max_adam_iters):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets).sum()
            loss.backward()
            optimizer.step()
        if verbose:
            print("Adam fallback finished")
        return True


"""
building a custom model for icm to enable Specific kernel configuration, Custom mean function:
MultitaskMean(ConstantMean()) — one mean per task, Likelihood control:
MultitaskGaussianLikelihood with custom noise constraints
"""


class ICMMultiOutputGP(ExactGP): 


    def __init__(self, train_X, train_Y, likelihood, rank: int = 1):
        """
        train_X : [N, d]  (inputs, normalized to [0,1]^d)
        train_Y : [N, L]  (per-edge (task) outputs / error on edges)
        """
        super().__init__(train_X, train_Y, likelihood)

        self.num_tasks = train_Y.shape[-1] 
        self.mean_module = MultitaskMean(
             ConstantMean(),  
             num_tasks=self.num_tasks)
        #self.mean_module = ConstantMean(batch_shape=torch.Size([self.num_tasks])) # gives one learnable mean per task 


        # Data kernel k(x, x') on OD inputs, shared over input.
        # No ScaleKernel: MultitaskKernel handles scaling (avoids overparameterization).
        data_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=train_X.shape[-1],
            lengthscale_constraint=Interval(0.005, 4.0),
        )

        # Kronecker of data kernel and task kernel : multitask kernel combine data kernel and task kernel
        self.covar_module = MultitaskKernel( 
            data_kernel,
            num_tasks=self.num_tasks,
            rank=rank, 
        )

    def forward(self, x):
        mean_x = self.mean_module(x) 
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


    def transform_inputs(self, X=None, **kwargs): # normalisation well be done outside the model 
        """
        BoTorch calls model.transform_inputs(X=...). For a pure GPyTorch model
        we simply return X unchanged.
        """
        return X


def initialize_icm_gp(
    train_X_norm: torch.Tensor,
    train_Y: torch.Tensor,
    verbose: bool = False,
    rank: int = 1,
):
    """
    Initialize icm with specified rank.

    train_X_norm : [N, d]  (normalized inputs in [0,1]^d)
    train_Y      : [N, L]  (per-edge errors / tasks)
    verbose      : bool    (print diagnostic information)

    return : 
    model : ICMMultiOutputGP
    mll   : ExactMarginalLogLikelihood
    """
    if train_X_norm.dim() != 2 or train_Y.dim() != 2:
        raise ValueError("Expected train_X_norm [N,d], train_Y [N,L].")    

    N, d = train_X_norm.shape
    N2, L = train_Y.shape
    if N != N2:
        raise ValueError(f"Mismatch in N: X has {N}, Y has {N2}.")   # check if the number of inputs and outputs are the same for debug 

    likelihood = MultitaskGaussianLikelihood(
        num_tasks=L,
        noise_constraint=Interval(1e-4, 1.0), # for diagonal noise matrix, This corresponds to a diagonal task-noise covariance, i.e., independent, task-specific observation noise modeled via a Kronecker product with the identity over inputs.
    )   # this is diagonal noise matrix,  made for identity matox knoecker product with task specific noise matrix, noise is indepednt across task 

    model = ICMMultiOutputGP(
        train_X=train_X_norm,
        train_Y=train_Y,
        likelihood=likelihood,
        rank=rank,
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll








class LinearAggregationICM(Model):   # linear aggregagtion to be used in ei 
    """
    Wrap an ICMMultiOutputGP (L outputs) and expose a scalar GP:

        S(x) = Σ_l w_l * e_l(x)

    where:
      - e(x) ∈ R^L are per-task / per-edge outputs of the ICM
      - w ∈ R^L are fixed uniform weights
    """

    def __init__(self, base_icm_model: ICMMultiOutputGP, weights: torch.Tensor, normalize_weights: bool = False):
        super().__init__()
        self.base_model = base_icm_model

        w = torch.as_tensor(
            weights,
            dtype=base_icm_model.train_targets.dtype,
            device=base_icm_model.train_targets.device,
        )
        if normalize_weights:
            w = w / w.sum()
        self.register_buffer("w", w.view(-1))  # [L]

    @property  # this is for the botorch model to know that the model has one output
    def num_outputs(self) -> int:
        # one scalar output S(x)
        return 1

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
        **kwargs,
    ) -> GPyTorchPosterior:
        """
        X : [b, d] or [b, 1, d] (q=1)

        For each x_b:
          - get multi-output dist e_b(x) (L tasks)
          - aggregate S_b = wᵀ e_b
            Var(S_b) = wᵀ Σ_b w
        """
        # Handle shapes [b,1,d] vs [b,d] : for the botorch model to know that the input has one dimension
        if X.dim() == 3:
            b, q, d = X.shape
            if q != 1: # no batch acquisition 
                #sequential BO: acquire 1 point → evaluate → update model → repeat
                raise RuntimeError("LinearAggregationICM supports q=1 only.")
            X = X.view(b, d)
        elif X.dim() == 2:
            b, d = X.shape
        else:
            raise RuntimeError(f"Unexpected X.dim()={X.dim()}, expected 2 or 3.")

        
        self.base_model.eval()
        self.base_model.likelihood.eval() 

      
        multi_dist = self.base_model(X)   # MultitaskMultivariateNormal
        if observation_noise:
            multi_dist = self.base_model.likelihood(multi_dist)


        mean_e = multi_dist.mean                       # [b, L], Posterior mean of the latent function for each task, expected value of task l at point i (noise-free)
        cov_e = multi_dist.covariance_matrix           # b*L, b*L

        L = self.base_model.num_tasks
        w = self.w 

        mus = []
        vars_ = []

        for i in range(b): # for each point in the batch
            start = i * L
            end   = (i + 1) * L

            mean_i = mean_e[i]                  # L
            cov_i  = cov_e[start:end, start:end]  # L, L

            mu_S  = (mean_i * w).sum()          # wᵀ μ
            var_S = w @ cov_i @ w               # wᵀ Σ w, as an exemple if we habe 2 tasks : var_S = w1^2 * var_1 + w2^2 * var_2 + 2 * w1 * w2 * cov_12


            if var_S.item() < -1e-12:
                raise RuntimeError(f"Negative scalar variance encountered: {var_S.item()}")

            # treat tiny negatives as numerical zero
            var_S = torch.clamp(var_S, min=0.0)

      
            mus.append(mu_S)
            vars_.append(var_S)

        mus   = torch.stack(mus).view(b, 1)      # b, 1
        vars_ = torch.stack(vars_).view(b, 1, 1) # b, 1, 1

        mvn = MultivariateNormal(mus, vars_)
       #expects a covariance matrix of shape (b, 1, 1), i.e. b independent 1×1 covariance matrices
        return GPyTorchPosterior(mvn)


