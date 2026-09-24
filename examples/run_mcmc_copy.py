import gaiaDR4_pipeline as gp
from pathlib import Path
from tqdm import tqdm 
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import argparse 
import arviz as az


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobID", type=int)
    parser.add_argument("-a", "--arrayID", type=int)

    args = parser.parse_args()

    jobID = args.jobID
    arrayID = args.arrayID

    cwd = Path("/work/hdd/bfoc/nm78/mstar_dist_480")

    source_ids = np.loadtxt(cwd / "source_ids.txt", dtype=str)

    source_id = int(source_ids[arrayID])
    t = Table.read(f"/work/hdd/bfoc/nm78/gaiamock_data/EpochAstrometry-Gaia_DR4_{source_id}.fits")
    mstar = t.meta["MSTAR"]
    ra, dec = t.meta["RA"], t.meta["DEC"]

    t_ast_yr, psi, plx_factor, ast_obs, ast_err = t['obs_time'].data, t['scan_angle'].data, t['parallax_factor'].data, t['along_scan_obs'].data, t['along_scan_error'].data
    t_binned, w_binned, sig_w_binned, pf_binned, psi_binned = t_ast_yr * 365.25, ast_obs * 1e-3, ast_err * 1e-3, plx_factor, psi
    t_binned = np.array(t_binned, dtype=np.float64)
    w_binned = np.array(w_binned, dtype=np.float64)
    sig_w_binned = np.array(sig_w_binned, dtype=np.float64)
    pf_binned = np.array(pf_binned, dtype=np.float64)
    psi_binned = np.array(psi_binned, dtype=np.float64)

    save_dir = cwd / f"mcmc_files/{source_id}" 
    for d in ['idata', 'csv', 'plots', 'out']:
        Path(save_dir / d).mkdir(parents=True, exist_ok=True)

    print(f"Starting run {jobID}...")

    # Initial RUWE check
    ruwe, mu, sigma_mu = gp.utils.check_ruwe(t_ast_yr = t_binned/365.25, psi = psi_binned, plx_factor = pf_binned, ast_obs = w_binned * 1e3, ast_err = sig_w_binned * 1e3)
    print(f'RUWE: {ruwe}')

    single_model = gp.mcmc.build_single_star_model(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned)
    binary_model = gp.mcmc.build_planet_star_model_campbell_TI(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned)

    single_initvals, binary_initvals, metadata = gp.utils.generate_initvals_from_p0(single_model, binary_model, source_id, cwd, mstar)

    print("\nFitting Single Star Model...")
    idata_single = gp.mcmc.run_mcmc(single_model, initvals=single_initvals, init='adapt_full', progressbar=False, callback=gp.mcmc.make_progress_callback(1000, 1000, 4), random_seed=42)
    print("\nFitting Planet-Star Model...")
    idata_binary = gp.mcmc.run_mcmc(binary_model, initvals=binary_initvals, target_accept=0.9, tune=2000, draws=2000, cores=4, chains=4, init='adapt_full', random_seed=42, progressbar=False, callback=gp.mcmc.make_progress_callback(2000, 2000, 4))
    az.to_netcdf(idata_binary, "%s/idata/binary_%s.nc"%(save_dir, jobID))
    az.to_netcdf(idata_single, "%s/idata/single_%s.nc"%(save_dir, jobID))

    gp.plotting.plot_model_fits(t_binned, w_binned, sig_w_binned, idata_single, idata_binary, save_fn = save_dir / f"plots/model_comparison_fits_{jobID}.png")

    loo_single, loo_binary, comp_df = gp.mcmc.compare_models_loo(idata_single, idata_binary)
    comp_df.to_csv('%s/csv/model_comparison_results_%s.csv'%(save_dir, jobID))

    print('Pipeline run complete.')

    # gp.mcmc.run_pipeline(cwd, jobID, source_id, t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, mstar, in_notebook=False)