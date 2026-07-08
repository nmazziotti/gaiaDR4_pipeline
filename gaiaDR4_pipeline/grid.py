import torch
import numpy as np
from pathlib import Path
import time

def solve_Kepler_equation_torch(M, ecc, xtol=1e-10, max_iter=100):
    """
    Vectorized Kepler solver in PyTorch for tensors of shape (N, n_obs)
    Solves: E - ecc * sin(E) = M
    """
    # Initial guess
    E = M.clone() 
    for _ in range(max_iter):
        f = E - ecc * torch.sin(E) - M
        f_prime = 1.0 - ecc * torch.cos(E)
        delta = f / f_prime
        E -= delta
        
        # Check convergence (optional, but keeps it fast)
        if torch.max(torch.abs(delta)) < xtol:
            break
    return E

def get_chi2_astrometry_vectorized_torch(n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, P, phi_p, ecc, device="cuda"):
    N = len(P)
    dtype = torch.float64 
    
    # Ensure proper data types on GPU
    t_ast_yr   = t_ast_yr.to(device=device, dtype=dtype)
    psi        = psi.to(device=device, dtype=dtype)
    plx_factor = plx_factor.to(device=device, dtype=dtype)
    ast_obs    = ast_obs.to(device=device, dtype=dtype)
    ast_err    = ast_err.to(device=device, dtype=dtype)
    P          = P.to(device=device, dtype=dtype)
    phi_p      = phi_p.to(device=device, dtype=dtype)
    ecc        = ecc.to(device=device, dtype=dtype)

    # --- Inverse Covariance Diagonal ---
    ivar = 1.0 / (ast_err**2)
    Cinv = torch.diag(ivar) # (n_obs, n_obs)

    # --- Expand arrays for broadcasting ---
    t    = t_ast_yr[None, :]   # (1, n_obs)
    P_   = P[:, None]          # (N, 1)
    phi_ = phi_p[:, None]      # (N, 1)
    ecc_ = ecc[:, None]        # (N, 1)
    psi_ = psi[None, :]        # (1, n_obs)
    plx_ = plx_factor[None, :] # (1, n_obs)

    # --- Mean anomaly and Kepler solve ---
    M = 2 * torch.pi * t * 365.25 / P_ - phi_   # (N, n_obs)
    E = solve_Kepler_equation_torch(M, ecc_)    # (N, n_obs) 
    
    X = torch.cos(E) - ecc_                     # (N, n_obs)
    Y = torch.sqrt(1 - ecc_**2) * torch.sin(E)  # (N, n_obs)

    # --- Build Design Matrix A: shape (N, n_obs, 9) ---
    A = torch.zeros((N, n_obs, 9), device=device, dtype=dtype)
    
    # FIXED: Exactly mirroring your original NumPy equations
    A[:, :, 0] = torch.sin(psi_)
    A[:, :, 1] = t * torch.sin(psi_)
    A[:, :, 2] = torch.cos(psi_)
    A[:, :, 3] = t * torch.cos(psi_)
    A[:, :, 4] = plx_
    A[:, :, 5] = X * torch.sin(psi_)
    A[:, :, 6] = Y * torch.sin(psi_)
    A[:, :, 7] = X * torch.cos(psi_)
    A[:, :, 8] = Y * torch.cos(psi_)

    # --- Batched Linear Algebra ---
    At = A.permute(0, 2, 1)  # (N, 9, n_obs)

    AtCinv  = At @ Cinv        # (N, 9, n_obs)
    AtCinvA = AtCinv @ A       # (N, 9, 9)
    AtCinvY = (AtCinv @ ast_obs[:, None]) # (N, 9, 1)

    # Safe fast solve
    mu = torch.linalg.solve(AtCinvA, AtCinvY).squeeze(-1)  # (N, 9)

    # --- Chi2 Calculation ---
    Lambda_pred = (A @ mu[:, :, None]).squeeze(-1)  # (N, n_obs)
    residuals   = Lambda_pred - ast_obs[None, :]    # (N, n_obs)
    chi2 = torch.sum(residuals**2 / ast_err**2, dim=1)  # (N,)

    return torch.column_stack([chi2, mu])

def grid_search_pure_torch(t_ast_yr, psi, plx_factor, ast_obs, ast_err, P_range=[10, 1e4], use_logspace=True, device="cuda", gpu_chunk_size=100000, N_logP = 1800, N_ecc=60,  N_phi=60):
    """
    100% Pure PyTorch Grid Search. 
    Accepts Python iterables or Tensors, keeps EVERYTHING on the GPU, 
    and returns pure PyTorch Tensors.
    """
    dtype = torch.float64  # Double precision for high-accuracy astrometry linear algebra

    # --- 1. Load/Convert Observational Data directly to GPU ---
    t_ast_yr   = torch.as_tensor(t_ast_yr, dtype=dtype, device=device)
    psi        = torch.as_tensor(psi, dtype=dtype, device=device)
    plx_factor = torch.as_tensor(plx_factor, dtype=dtype, device=device)
    ast_obs    = torch.as_tensor(ast_obs, dtype=dtype, device=device)
    ast_err    = torch.as_tensor(ast_err, dtype=dtype, device=device)
    n_obs      = len(t_ast_yr)

    # --- 2. Build Search Grid Directly on the GPU Memory ---
    if use_logspace:
        log_P_grid = torch.linspace(torch.log(torch.tensor(P_range[0])), torch.log(torch.tensor(P_range[1])), N_logP, dtype=dtype, device=device)
        P_grid = torch.exp(log_P_grid)
    else:
        P_grid = torch.linspace(P_range[0], P_range[1], N_logP, dtype=dtype, device=device)
        
    phi_grid = torch.linspace(-torch.pi, torch.pi, N_phi, dtype=dtype, device=device)
    ecc_grid = torch.linspace(0, 0.9999, N_ecc, dtype=dtype, device=device)

    # Replaces itertools.product natively on CUDA
    P_mesh, ecc_mesh, phi_mesh = torch.meshgrid(P_grid, ecc_grid, phi_grid, indexing="ij")
    
    # Flatten the parameters into long parallel arrays (6,480,000 elements)
    P_samples   = P_mesh.reshape(-1)
    ecc_samples = ecc_mesh.reshape(-1)
    phi_samples = phi_mesh.reshape(-1)
    log_P_samples = torch.log(P_samples)
    
    N_total = len(P_samples)

    # --- 3. Run Batched Processing Loop over the GPU ---
    chi2_list = []
    mu_list = []

    for i in range(0, N_total, gpu_chunk_size):
        end_idx = min(i + gpu_chunk_size, N_total)
        
        P_chunk   = P_samples[i:end_idx]
        phi_chunk = phi_samples[i:end_idx]
        ecc_chunk = ecc_samples[i:end_idx]

        # Process chunk natively in parallel via PyTorch
        res_chunk = get_chi2_astrometry_vectorized_torch(
            n_obs, t_ast_yr, psi, plx_factor, ast_obs, ast_err, 
            P_chunk, phi_chunk, ecc_chunk, device=device
        )
        
        chi2_list.append(res_chunk[:, 0])
        mu_list.append(res_chunk[:, 1:])

    # Concatenate the results on the GPU
    chi2_samples = torch.cat(chi2_list, dim=0)
    mu_samples   = torch.cat(mu_list, dim=0)

    # --- 4. GPU-Vectorized Log-Likelihood & Weights ---
    constant_term = torch.sum(torch.log(2 * torch.pi * ast_err**2))
    log_L = -0.5 * chi2_samples - 0.5 * constant_term
    
    # Log-Sum-Exp safety shift to avoid numeric underflow/overflow
    log_L -= torch.max(log_L)
    L = torch.exp(log_L)
    weights = L / torch.sum(L)

    # --- 5. Return everything as active GPU Tensors ---
    return log_P_samples.cpu().numpy(), ecc_samples.cpu().numpy(), phi_samples.cpu().numpy(), mu_samples.cpu().numpy(), chi2_samples.cpu().numpy(), weights.cpu().numpy(), log_P_grid.cpu().numpy(), ecc_grid.cpu().numpy(), phi_grid.cpu().numpy()

def compute_marginal_weights(samples, chi2_samples, weights):
    unique_vals = np.unique(samples)
    marginal_weights = np.array([
        weights[samples == val].sum() 
        for val in unique_vals
    ])

    min_chi2 = np.array([
        np.min(chi2_samples[samples == val]) 
        for val in unique_vals
    ])

    return unique_vals, marginal_weights/ np.sum(marginal_weights), min_chi2

def run_grid_search_torch(cwd, source_id, t_ast_yr_raw, psi_raw, plx_factor_raw, ast_obs_raw, ast_err_raw, device="cuda", **kwargs):
    # 1. Verify CUDA is active

    # 4. Use torch.from_numpy to automatically and safely fix the byte-order 
    # during the transfer directly onto the GPU
    dtype = torch.float64
    t_ast_yr   = torch.from_numpy(t_ast_yr_raw).to(device=device, dtype=dtype)
    psi        = torch.from_numpy(psi_raw).to(device=device, dtype=dtype)
    plx_factor = torch.from_numpy(plx_factor_raw).to(device=device, dtype=dtype)
    ast_obs    = torch.from_numpy(ast_obs_raw).to(device=device, dtype=dtype)
    ast_err    = torch.from_numpy(ast_err_raw).to(device=device, dtype=dtype)
    # =========================================================================

    # print("Start execution on GPU...")
    start = time.time()
    
    # Notice we dropped the 'cores' parameter entirely! The GPU handles parallelization natively.
    log_P_samples, ecc_samples, phi_samples, mu_samples, chi2_samples, weights, log_P_grid, ecc_grid, phi_grid = grid_search_pure_torch(
        t_ast_yr, psi, plx_factor, ast_obs, ast_err, device=device, **kwargs)

    end = time.time() - start
    # print("Done") 
    # print(f"End: Took {end:.3f} seconds")

    unique_logP, marginal_weights_logP, min_chi2_logP = compute_marginal_weights(log_P_samples, chi2_samples, weights)
    save_dir = cwd / f"grid_search_files/{source_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
    save_dir / 'grid_search.npz',
    log_P=unique_logP,
    min_chi2=min_chi2_logP
    )

    global_min_chi2_idx = np.argmin(chi2_samples)
    log_P, ecc, phi = log_P_samples[global_min_chi2_idx], ecc_samples[global_min_chi2_idx], phi_samples[global_min_chi2_idx]
    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_samples[global_min_chi2_idx, :]
    p0 = np.array([log_P, ecc, phi, ra_off, pmra, dec_off, pmdec, plx, B, G, A, F])
    np.savetxt(save_dir / 'grid_search_p0.txt', p0)

def grid_search_already_ran(sourceID):
    if Path(f'/work/hdd/bfoc/nm78/standard_pipeline/runs/{sourceID}/optimize/grid_search_p0.txt').exists():
        return True
    else:
        return False 
