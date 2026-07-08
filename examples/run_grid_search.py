import gaiaDR4_pipeline as gp
from pathlib import Path
from tqdm import tqdm 
import os
import torch

#sourceIDs = find_file(tag="SingleStar_M035", sourceID_only=True)
source_ids = [3926186255616949504, 4181040337841125632]

for source_id in tqdm(source_ids):
    t_ast_days, psi_raw, ast_obs_raw, ast_err_raw, plx_factor_raw = gp.utils.extract_time_series(source_id)
    t_ast_yr_raw = t_ast_days / 365.25

    cwd = Path(os.getcwd()) / "dr4_prerelease"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gp.grid.run_grid_search_torch(cwd, source_id, t_ast_yr_raw, psi_raw, plx_factor_raw, ast_obs_raw * 1e3, ast_err_raw * 1e3, device=device)
