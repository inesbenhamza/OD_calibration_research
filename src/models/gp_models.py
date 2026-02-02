# Third-party imports
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


def set_covar_module(kernel: str, dim: int):
    if kernel == "matern-1p5":
        return ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
    elif kernel == "matern-2p5":
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
    elif kernel == "rbf":
        return ScaleKernel(RBFKernel(ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")





#scalar gp
def initialize_vanillabo_model(train_X, train_Y, kernel: str):
    
    dim = train_X.size(-1)
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = set_covar_module(kernel, dim)
    
    model = SingleTaskGP(
        train_X,
        train_Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    


    return model







# INDEPENDENT GP MODELS
def initialize_independent_gp_models_with_modellist(train_X, train_Y_multi, kernel: str = "matern-2p5"):
    """
    Fit independent GPs for each edge/link output using ModelListGP.

    Parameters
    ----------
    train_X : torch.Tensor
        (n_observations, n_od_pairs) - OD configurations
    train_Y_multi : torch.Tensor
        (n_observations, n_edges) - Traffic counts for each edge
    kernel : str, optional
        Kernel type ("matern-1p5", "matern-2p5", "rbf"). Default: "matern-2p5"

    Returns
    -------
    tuple
        (model_list_gp, mlls) - ModelListGP containing all edge GPs and list of MLLs
    """
    n_edges = train_Y_multi.size(1)
    dim = train_X.size(1)  # n_od_pairs

    individual_models = []
    mlls = []

    for edge_idx in range(n_edges):
        
        train_Y_edge = train_Y_multi[:, edge_idx].unsqueeze(-1)  # [N, 1]
        
        # Filter out NaN values for this specific edge
        # Each edge GP trains only on samples where that edge has valid (non-NaN) error
        valid_mask = ~torch.isnan(train_Y_edge).squeeze(-1)  # [N]
        #boolean tensor where True means "this value is a valid number" and False means "this value is NaN".
        
        
        #mask.sum(): In PyTorch, summing a boolean tensor treats True as 1 and False as 0.
        if valid_mask.sum() == 0: #meaning all are nan as its all false 
            # All samples have NaN for this edge - this shouldn't happen in practice
            # but we'll use all samples and let GPyTorch handle it
            train_X_edge = train_X
            train_Y_edge_filtered = train_Y_edge
        else:
            train_X_edge = train_X[valid_mask] # nouveau tenseur train_X_edge qui ne contient que les éléments de train_X où la valeur correspondante dans valid_mask est True.
            train_Y_edge_filtered = train_Y_edge[valid_mask]

        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = set_covar_module(kernel, dim)
        gp_model = SingleTaskGP(
            train_X_edge,
            train_Y_edge_filtered,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1)
        )

        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

        individual_models.append(gp_model)
        mlls.append(mll)

    # Wrap all individual GPs in a ModelListGP
    model_list_gp = ModelListGP(*individual_models)

    return model_list_gp, mlls




def predict_independent_gps(model_list_gp, test_X):
  
    predictions = []
    uncertainties = []

    # Access individual GPs from ModelListGP
    for gp_model in model_list_gp.models:
        gp_model.eval()

        with torch.no_grad():
            posterior = gp_model.posterior(test_X)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            predictions.append(mean)
            uncertainties.append(variance.sqrt())

    predictions = torch.stack(predictions, dim=-1)
    uncertainties = torch.stack(uncertainties, dim=-1)

    return predictions, uncertainties



# LINEAR AGGREGATION MODEL


def make_linear_aggregation_model_from_error_gps(model_list_gp_errors, weights=None, ground_truth_flows=None, expect_flow_weights: bool = False):
    
    L = len(model_list_gp_errors.models)  # number of edges
    
    # Infer dtype from the first GP model (for consistency with GP models)
    # Default to double precision for numerical stability
    if len(model_list_gp_errors.models) > 0:
        model_dtype = next(model_list_gp_errors.models[0].parameters()).dtype
    else:
        model_dtype = torch.double  

    if weights is None: 
        # No weights provided - calculate based on ground_truth_flows or use uniform
        if ground_truth_flows is not None:
            # For cubic metric: use cubic weight formula
            flows = torch.as_tensor(ground_truth_flows, dtype=model_dtype)
            w = flows / (L * flows.sum())
            print("Using cubic weights (calculated from flows)")
        else:
            # Default: uniform weights for squared error metric
            w = torch.ones(L, dtype=model_dtype) / L
            print(f"Using uniform weights (1/{L} each)")
    else: # Use provided weights directly 
        if isinstance(weights, torch.Tensor):
            w = weights.to(dtype=model_dtype)
        else:
            w = torch.as_tensor(weights, dtype=model_dtype)
        is_uniform = torch.allclose(w, torch.full_like(w, 1.0/L), atol=1e-6)
        print(f"Using provided weights (sum={w.sum().item():.6f}, scale={w.sum().item() * L:.6f}, expect_flow_weights={expect_flow_weights})")
        print(f"  Weight range: [{w.min().item():.6f}, {w.max().item():.6f}]")
        print(f"  First 5 weights: {w[:5].numpy()}")
        if expect_flow_weights:
            if is_uniform:
                print("  WARNING: Expected flow-proportional weights, but got uniform.")
            else:
                print("  Weights are flow-proportional (non-uniform)")

    class LinearAggregationModel(ModelListGP):
        def __init__(self, *models, w): # call the model using model = linear_aggregation_model(gp1, gp2, gp3, w)
            super().__init__(*models)# call the parent class ModelListGP 
            # its equvalent to :
            #  ModelListGP.__init__(self, gp1, gp2, gp3, ..., gp51)
            # This creates: self.models = [gp1, gp2, gp3, ..., gp51]
            self.register_buffer("w", w)

        @property
        def num_outputs(self):
            """Override to indicate this is a single-output model."""
            return 1

        def posterior(self, X, observation_noise=False, **kwargs):
            """
            Compute posterior for S(x) = sum_l [w_l * e_l(x)]

            Returns
            -------
            GPyTorchPosterior with:
            - mean: (batch_size,) or (batch_size, 1)
            - covariance: (batch_size, 1, 1) diagonal
            """
            # collecting posteriors from each per-edge-error GP
            posts = [m.posterior(X, observation_noise=observation_noise) for m in self.models]
            means = torch.stack([p.mean.squeeze(-1) for p in posts], dim=-1)     # (b, L)
            vars_ = torch.stack([p.variance.squeeze(-1) for p in posts], dim=-1) # (b, L)

            # linear aggregation is allowed because it's a linear combination of gaussian random variables
            # Any linear combination of jointly Gaussian random variables is Gaussian.
            mu_S = (means * self.w).sum(dim=-1)  # (b,)
            var_S = (vars_ * (self.w ** 2)).sum(dim=-1).clamp_min(1e-12)  # (b,)

            # Create MultivariateNormal with proper shapes
            # MVN expects: mean (b,), covariance_matrix (b, 1, 1) (botorch expects 3D for covariance)
            cov_matrix = var_S.view(-1, 1, 1)

            mvn = MultivariateNormal(mu_S, cov_matrix)
            return GPyTorchPosterior(mvn)

        def forward(self, X):
            """For compatibility with BoTorch"""
            return self.posterior(X)

    return LinearAggregationModel(*model_list_gp_errors.models, w=w)




def initialize_model(model_name: str, train_X, train_Y, kernel: str):
    """
    Factory function to initialize GP models based on model name.
    
    Parameters
    ----------
    model_name : str
        Model type: "vanillabo" or "independent_gp"/"ind_gp"
    train_X : torch.Tensor
        Training inputs
    train_Y : torch.Tensor
        Training targets (for vanillabo: aggregated error, for ind_gp: per-edge errors)
    kernel : str
        Kernel type ("matern-1p5", "matern-2p5", "rbf")
    
    Returns
    -------
    Model instance (SingleTaskGP for vanillabo, ModelListGP for ind_gp)
    """
    if model_name == "vanillabo":
        return initialize_vanillabo_model(train_X, train_Y, kernel)
    elif model_name in ["ind_gp", "independent_gp"]:
        return initialize_independent_gp_models_with_modellist(train_X, train_Y, kernel)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: 'vanillabo', 'independent_gp', 'ind_gp'")

