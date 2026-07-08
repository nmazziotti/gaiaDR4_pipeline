import argparse 
import gaiaDR4_pipeline as gp 
from pathlib import Path 
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobID", type=int)
    parser.add_argument("-a", "--arrayID", type=int)

    args = parser.parse_args()

    jobID = args.jobID
    arrayID = args.arrayID
    version='dr4'

    sourceIDs = [1457486023639239296, 3937211745905473024]
    mstars = [0.74, 1.046]

    sourceID = sourceIDs[arrayID]
    mstar = mstars[arrayID]

    base_dir = Path('/work/hdd/bfoc/nm78/dr4_prerelease/')

    t_binned, psi_binned, w_binned, sig_w_binned, pf_binned = gp.utils.extract_time_series(sourceID)
     
    gp.mcmc.run_pipeline(base_dir, jobID, str(sourceID), t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, mstar)