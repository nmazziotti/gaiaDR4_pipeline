import gaiaDR4_pipeline as gp
from pathlib import Path
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import argparse 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobID", type=int)
    parser.add_argument("-a", "--arrayID", type=int)

    args = parser.parse_args()

    jobID = args.jobID
    arrayID = args.arrayID

    cwd = Path(os.getcwd()) / "dr4_prerelease"
    table = Table.read(cwd / "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml", format="votable")
    df_data = table.to_pandas()

    source_ids = [1457486023639239296, 3937211745905473024]
    mstars = [0.74, 1.046]

    source_id = source_ids[arrayID]
    mstar = mstars[arrayID]

    t_binned, psi_binned, w_binned, sig_w_binned, pf_binned = gp.utils.extract_time_series(source_id, df_data)
    ra, dec = gp.utils.extract_ra_dec(source_id, df_data)

    gp.mcmc.run_pipeline(cwd, jobID, source_id, t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, mstar, in_notebook=False)