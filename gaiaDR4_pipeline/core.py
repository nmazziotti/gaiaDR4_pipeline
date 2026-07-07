import arviz as az
import time 
from datetime import datetime
from pathlib import Path
import pandas as pd 
import argparse
import astropy
import numpy as np 

# from . import utils
# from . import modeling

import utils
import modeling

def run_pipeline(output_dir, jobID, sourceID, t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, mstar):
    begin = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    start_pipeline = time.time()

    save_dir = output_dir / f"mcmc_files/{sourceID}" 
    # for d in ['idata', 'csv', 'plots', 'out']:
    #     Path(save_dir / d).mkdir(parents=True, exist_ok=True)
  
    #Path(output_dir / f"timestamps/{jobID}").mkdir(parents=True, exist_ok=True)
    
    print(f"Starting run {jobID}...")

    # Initial RUWE check
    ruwe, mu, sigma_mu = utils.check_ruwe(t_ast_yr = t_binned/365.25, psi = psi_binned, plx_factor = pf_binned, ast_obs = w_binned * 1e3, ast_err = sig_w_binned * 1e3)
    print(f'RUWE: {ruwe}')

    # Build models
    single_model = modeling.build_single_star_model(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned)
    binary_model = modeling.build_planet_star_model_campbell_TI(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, angles='uniform', unit_disk=False)

    start_map = time.time()

    single_initvals, binary_initvals = utils.generate_initvals_from_p0(single_model, binary_model, sourceID, output_dir, mstar)
    print(binary_initvals)
    # single_initvals, binary_initvals, metadata = generate_initvals_from_grid_search(single_model, binary_model, t_ast_yr, psi, plx_factor, ast_obs, ast_err, t.meta['MSTAR'], cores=4)
    # generate_initvals(single_model, binary_model, t_ast_yr, psi, plx_factor, ast_obs, ast_err, mstar, c_funcs)
    map_time = time.time() - start_map
    print(f"MAP optimization took: {map_time:.2f} seconds")


    # Fit models and get inference data
    print("\nFitting Single Star Model...")
    idata_single = modeling.run_mcmc(single_model, initvals=single_initvals, init='adapt_full', progressbar=False, callback=modeling.make_progress_callback(1000, 1000, 4), random_seed=42)
    print("\nFitting Planet-Star Model...")
    idata_binary = modeling.run_mcmc(binary_model, initvals=binary_initvals, target_accept=0.9, tune=2000, draws=2000, cores=4, chains=4, init='adapt_full', random_seed=42, progressbar=False, callback=modeling.make_progress_callback(2000, 2000, 4))
    az.to_netcdf(idata_binary, "%s/idata/binary_%s.nc"%(save_dir, jobID))
    az.to_netcdf(idata_single, "%s/idata/single_%s.nc"%(save_dir, jobID))

    # Compare models using LOO
    loo_single, loo_binary, comp_df = modeling.compare_models_loo(idata_single, idata_binary)

    # Plot comparison
    modeling.plot_model_fits(t_binned, w_binned, sig_w_binned, idata_single, idata_binary, save_dir, jobID)

    # Save results
    print(f"\nSaving results...")
    comp_df.to_csv('%s/csv/model_comparison_results_%s.csv'%(save_dir, jobID))

    pipeline_time = time.time() - start_pipeline
    print(f"Pipeline runtime: {pipeline_time/60:.2f} minutes")

    print(f"Sampling time: {idata_binary.sample_stats.sampling_time:.2f} seconds")
    print(f"Total fitting time for binary model: {(idata_binary.sample_stats.sampling_time + map_time)/60:.2f} minutes")

    timing_dict = {'Binary_Optimization': map_time, 'Single_Sampling': idata_single.sample_stats.sampling_time, 'Binary_Sampling': idata_binary.sample_stats.sampling_time, 'Total_Runtime': pipeline_time}
    timing_df = pd.DataFrame(timing_dict, index=[0])
    timing_df.to_csv('%s/csv/timing_%s.csv'%(save_dir, jobID), index=False)

    end = datetime.now().strftime("%a %b %d %H:%M:%S %Y")
    print('Pipeline run complete.')

def extract_time_series(source_id):
    table = astropy.table.Table.read("/work/hdd/bfoc/nm78/dr4_prerelease/extracted_files/GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml", format="votable")
    df = table.to_pandas()

    mask = df['source_id'] == source_id

    def bin(column):
        col = np.stack(df.loc[mask][column].values) 
        #mean = col.mean(axis=1) 
        mean = np.ma.median(col, axis=1)  
        #return mean.data
        return mean.filled(np.nan)

    t_binned = bin('obs_time_tcb')
    psi_binned = np.deg2rad(bin('scan_pos_angle'))
    w_binned = bin('centroid_pos_al') * 1e-3
    sig_w_binned = bin('centroid_pos_error_al') * 1e-3
    pf_binned = df.loc[mask]['parallax_factor_al'].values

    idx = np.argsort(t_binned)        # ascending
    t_binned = t_binned[idx]
    psi_binned = psi_binned[idx]
    w_binned = w_binned[idx]
    sig_w_binned = sig_w_binned[idx]
    pf_binned = pf_binned[idx]
    

    is_nan = np.isnan(w_binned)
    print(f"NOTE: {np.sum(is_nan)/len(is_nan)*100:.2f}% of data points are NaN")
    t_binned = t_binned[~is_nan]
    psi_binned = psi_binned[~is_nan]
    w_binned = w_binned[~is_nan]
    sig_w_binned = sig_w_binned[~is_nan]
    pf_binned = pf_binned[~is_nan]

    tref = 2457936.875
    t_binned = t_binned / 1e9 / 86400 + 2455197.5 - tref

    return t_binned, psi_binned, w_binned, sig_w_binned, pf_binned

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

    t_binned, psi_binned, w_binned, sig_w_binned, pf_binned = extract_time_series(sourceID)
     
    run_pipeline(base_dir, jobID, str(sourceID), t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, mstar)