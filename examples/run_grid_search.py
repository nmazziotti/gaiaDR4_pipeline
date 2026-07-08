import gaiaDR4_pipeline as gp
from pathlib import Path
import os
import torch
import numpy as np
from astropy.table import Table
import argparse 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arrayID", type=int)

    args = parser.parse_args()
    arrayID = args.arrayID

    cwd = Path(os.getcwd()) / "dr4_prerelease"

    table = Table.read(cwd / "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml", format="votable")
    df_data = table.to_pandas()

    source_ids = [1457486023639239296, 3937211745905473024]
    source_id = source_ids[arrayID]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    t_binned, psi_binned, w_binned, sig_w_binned, pf_binned = gp.utils.extract_time_series(source_id, df_data)
    gp.grid.run_grid_search(cwd, source_id, t_binned, psi_binned, pf_binned, w_binned, sig_w_binned, 
                                device=device, N_logP=1800, N_ecc=60, N_phi=60, P_range=[10,1e4])
