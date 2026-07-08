import arviz as az
import corner
import exoplanet as xo
import matplotlib.pyplot as plt
import numpy as np 
import pymc as pm 
import pymc_ext as pmx
import pytensor.tensor as pt
import time 
from datetime import datetime
import pandas as pd 

from pymc.progress_bar import ProgressBarManager
import pymc.sampling.parallel as ps
from rich.console import Console
ps.Console = lambda theme=None: Console(theme=theme, force_terminal=True)

yr = 365.25

from . import utils, plotting

# MCMC Logging 
def _print_bars(chain_progress, chain_step_size, chain_divergences,
                chain_start_time, chain_elapsed, chain_count, total, bar_width):
    lines = []
    for chain_idx in sorted(chain_progress):
        completed = chain_progress[chain_idx]
        frac = min(completed / total, 1.0)
        filled = int(bar_width * frac)
        bar = "█" * filled + "-" * (bar_width - filled)
        pct = int(frac * 100)
        step = chain_step_size[chain_idx]
        divs = chain_divergences[chain_idx]

        start = chain_start_time[chain_idx]
        elapsed = chain_elapsed[chain_idx]
        if start is not None and completed > 0:
            rate = completed / max(elapsed, 1e-6)
            remaining = (total - completed) / rate if rate > 0 else 0
            elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
            remaining_str = f"{int(remaining//60):02d}:{int(remaining%60):02d}"
        else:
            elapsed_str = "00:00"
            remaining_str = "??:??"

        lines.append(
            f"Chain {chain_idx}: [{bar}] {pct:3d}% ({completed}/{total})"
            f" | step={step:.4f} | div={divs}"
            f" | elapsed={elapsed_str} | remaining={remaining_str}"
        )
    print("\n".join(lines) + "\n", flush=True)

def make_progress_callback(total_draws, tune, chain_count, bar_width=40):
    chain_progress = {i: 0 for i in range(chain_count)}
    chain_divergences = {i: 0 for i in range(chain_count)}
    chain_step_size = {i: 0.0 for i in range(chain_count)}
    chain_start_time = {i: None for i in range(chain_count)}
    chain_elapsed = {i: 0.0 for i in range(chain_count)}
    total = total_draws

    def callback(trace, draw):
        if draw.tuning:
            return

        if chain_start_time[draw.chain] is None:
            chain_start_time[draw.chain] = time.time()

        chain_progress[draw.chain] = draw.draw_idx - tune + 1

        if draw.stats:
            stats = draw.stats[0]
            if 'step_size' in stats:
                chain_step_size[draw.chain] = stats['step_size']
            if 'diverging' in stats and stats['diverging']:
                chain_divergences[draw.chain] += 1

        start = chain_start_time[draw.chain]
        if start is not None:
            chain_elapsed[draw.chain] = time.time() - start

        if draw.draw_idx % 100 == 0:
            _print_bars(chain_progress, chain_step_size, chain_divergences,
                       chain_start_time, chain_elapsed, chain_count, total, bar_width)

    def print_final():
        # Force all chains to 100%
        for i in range(chain_count):
            chain_progress[i] = total
        _print_bars(chain_progress, chain_step_size, chain_divergences,
                   chain_start_time, chain_elapsed, chain_count, total, bar_width)

    callback.print_final = print_final
    return callback

# Building models
def single_star_base(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, add_jitter=False):
    with pm.Model() as single_model:
        # Stellar position offset
        delta_ra = pm.Normal('delta_ra', mu=0., sigma=1.) # as
        delta_dec = pm.Normal('delta_dec', mu=0., sigma=1.) # as

        # delta_ra = pm.Uniform('delta_ra', lower=-1e-3, upper=1e-3) # as
        # delta_dec = pm.Uniform('delta_dec', lower=-1e-3, upper=1e-3) # as
        
        # Stellar proper motion
        pm_ra = pm.Normal('pm_ra', mu=0., sigma=1) # as/yr
        pm_dec = pm.Normal('pm_dec', mu=0., sigma=1) # as/yr
        
        # Parallax
        log_parallax = pm.Normal('log_parallax', mu=np.log(1e-2), sigma=1) # as 
        parallax = pm.Deterministic('parallax', pt.exp(log_parallax))
        
        if add_jitter:
            # Astrometric jitter
            log_sigma = pm.Uniform('log_sigma', lower=np.log(1e-6), upper=np.log(1e-3))

        # Model prediction
        def single_star_as(t, delta_ra, delta_dec, pm_ra, pm_dec, parallax, pf, psi):
            # single_ra = pm.Deterministic('single_ra', pm_ra*t/yr + delta_ra + parallax*pf*pt.sin(psi))
            # single_dec = pm.Deterministic('single_dec', pm_dec*t/yr + delta_dec + parallax*pf*pt.cos(psi))

            return (pm_ra*t/yr+delta_ra)*pt.sin(psi) + (pm_dec*t/yr+delta_dec)*pt.cos(psi) + parallax*pf
        
        model_single = pm.Deterministic('model_single', single_star_as(t_binned, delta_ra, delta_dec, pm_ra, pm_dec, parallax, pf_binned, psi_binned))
    return single_model

def build_single_star_model(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, add_jitter=False):
    """Build the single star astrometric model"""
    print("Building single star model...")

    single_model = single_star_base(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, add_jitter=add_jitter)
    
    with single_model:
        model_single = single_model['model_single']
        
        # Likelihood
        if add_jitter:
            log_sigma = single_model['log_sigma']
            total_sigma = pt.sqrt(sig_w_binned**2 + pt.exp(log_sigma)**2)
        else:
            total_sigma = sig_w_binned
        logl = pm.Normal('logl', mu=model_single, sigma=total_sigma, observed=w_binned)

        # Pointwise log likelihood 
        # logl_pointwise = pm.Deterministic(
        #     'logl_pointwise',
        #     -0.5 * pt.log(2 * np.pi * total_sigma**2) - 
        #     0.5 * ((w_binned - model_single) / total_sigma)**2
        # )
    
    return single_model

def xo_unit_disk(name_x, name_y, **kwargs):
    """Unit disk constraint without initval"""
    kwargs.pop("initval", None)
    kwargs["lower"] = -1.0
    kwargs["upper"] = 1.0
    
    x1 = pm.Uniform(name_x, **kwargs)
    x2 = pm.Uniform(f"__{name_y}_unit_disk", **kwargs)
    
    norm = pt.sqrt(1 - x1**2)
    pm.Potential(f"__{name_y}_jacobian", pt.log(norm))
    return x1, pm.Deterministic(name_y, x2 * norm)

def build_planet_star_model_campbell_TI(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, add_jitter=False, init_vals=None, angles='uniform', unit_disk=False):
    """Build the planet-star astrometric model"""

    print("Building planet-star model with combination of Campbell and Thiele-Innes elements...")

    binary_model = single_star_base(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, add_jitter=add_jitter)
    
    with binary_model:
        binary_model.model_name = 'campbell_TI'

        # Orbital period 
        log_p = pm.Uniform('log_p', lower=np.log(1), upper=np.log(1e5)) # days
        p = pm.Deterministic("p", pt.exp(log_p))

        # Phase of periastron 
        if angles == 'uniform':
            phi = pm.Uniform("phi", lower=-pt.pi, upper=pt.pi)
        elif angles == 'VonMises':
            kappa = 1e-6
            phi   = pm.VonMises("phi",  mu=0.0, kappa=kappa)
        elif angles == 'pmx.angle':
            phi = pmx.angle("phi")

        # Time of periastron
        tp = pm.Deterministic("tp", phi * p / (2 * pt.pi) ) # days
        

        if unit_disk:
            h, k = xo_unit_disk('h', 'k')
            ecc = pm.Deterministic('ecc', pt.sqrt(h**2 + k**2))
            omega = pm.Deterministic('omega', pt.arctan2(h, k)) # radians
            Omega = pm.Uniform("Omega", lower=0.0, upper=2*pt.pi)
        else:
            # Eccentricity
            ecc = pm.Uniform("ecc", lower=0.0, upper=1.0)

            if angles == 'uniform':
                plus = pm.Uniform("plus", lower=-pt.pi, upper=pt.pi)
                minus = pm.Uniform("minus", lower=-pt.pi, upper=pt.pi)
            elif angles == 'VonMises':
                kappa = 1e-6
                plus = pm.VonMises("plus", mu=0.0, kappa=kappa)  
                minus = pm.VonMises("minus", mu=0.0, kappa=kappa) 
            elif angles == 'pmx.angle':
                plus = pmx.angle("plus")
                minus = pmx.angle("minus")

            # Longitude of ascending node (Omega) and argument of periastron (omega)
            omega = pm.Deterministic("omega", (plus - minus) % (2*pt.pi)) # radians 
            Omega = pm.Deterministic("Omega", (plus + minus) % (2*pt.pi)) # radians
        
        # Inclination
        cosi = pm.Uniform('cosi', lower=-1., upper=1.)
        incl = pm.Deterministic("incl", pt.arccos(cosi)) # radians
    

        # Compute a0 
        # Grav = 2.95912208286e-4 # AU^3 Msun^-1 days^-2 
        log_a0 = pm.Uniform("log_a0", lower=np.log(1e-8), upper=np.log(10)) # as
        a0 = pm.Deterministic("a0", pt.exp(log_a0))
        # pm.Deterministic("a0", (Grav/(4 * pt.pi**2))**(1/3) * mp/(mstar + mp)**(2/3) * p**(2/3) * binary_model['parallax']) # as

        # Thiele-Innes elements 
        A = a0 * (pt.cos(omega) * pt.cos(Omega) - pt.sin(omega) * pt.sin(Omega) * cosi)
        B = a0 * (pt.cos(omega) * pt.sin(Omega) + pt.sin(omega) * pt.cos(Omega) * cosi)
        F = -a0 * (pt.sin(omega) * pt.cos(Omega) + pt.cos(omega) * pt.sin(Omega) * cosi)
        G = -a0 * (pt.sin(omega) * pt.sin(Omega) - pt.cos(omega) * pt.cos(Omega) * cosi)

        A_in_mas = pm.Deterministic("A_in_mas", A*1e3)
        B_in_mas = pm.Deterministic("B_in_mas", B*1e3)
        F_in_mas = pm.Deterministic("F_in_mas", F*1e3)
        G_in_mas = pm.Deterministic("G_in_mas", G*1e3)

        def get_position(t, p, ecc, phi, A, B, F, G):
            n = 2*pt.pi / p  # mean motion
            M = n * t - phi # mean anomaly

            # Solve Kepler's equation for true anomaly f
            f = xo.orbits.keplerian.get_true_anomaly(M, ecc + pt.zeros_like(M))
            r = (1 - ecc**2) / (1 + ecc * pt.cos(f))

            # # Compute projected positions using Thiele-Innes
            X = r * pt.cos(f)
            Y = r * pt.sin(f)

            b_ra = A*X + F*Y
            b_dec = B*X + G*Y

            return b_ra, b_dec 
        
        def binary_as(t, psi, p, ecc, phi, A, B, F, G):
            b_ra, b_dec = get_position(t, p, ecc, phi, A, B, F, G)
            
            binary_ra = pm.Deterministic('binary_ra', b_ra)
            binary_dec = pm.Deterministic('binary_dec', b_dec)

            return b_ra*pt.cos(psi) + b_dec*pt.sin(psi)

        model_binary =  pm.Deterministic('model_binary', binary_as(t_binned, psi_binned, p, ecc, phi, A, B, F, G))
        
        model_single = binary_model['model_single']

        # Optionally add jitter 
        if add_jitter:
            log_sigma = binary_model['log_sigma']
            total_sigma = pt.sqrt(sig_w_binned**2 + pt.exp(log_sigma)**2)
        else:
            total_sigma = sig_w_binned
        
        # Likelihoods
        logl = pm.Normal('logl', mu=model_single + model_binary, sigma=total_sigma, observed=w_binned)


        if angles == 'pmx.angle':
            # Pointwise log likelihood 
            logl_pointwise = pm.Deterministic(
                'logl_pointwise',
                -0.5 * pt.log(2 * np.pi * total_sigma**2) - 
                0.5 * ((w_binned - (model_single + model_binary)) / total_sigma)**2
            )
        
        # For plotting orbit 
        t_max_plot = pm.Deterministic('t_max_plot', 1.1 * p)
        t_plot = pt.linspace(0, t_max_plot, 1000)
        b_ra, b_dec = get_position(t_plot, p, ecc, phi, A, B, F, G)
        ra_plot = pm.Deterministic('ra_plot', b_ra)
        dec_plot = pm.Deterministic('dec_plot', b_dec)

    return binary_model

# Fitting models 
def run_mcmc(model, **kwargs):
    with model:
        idata = pm.sample(**kwargs)
        
        # Compute log likelihood
        avail_vars = [var for var in model.named_vars]

        if 'logl_pointwise' in avail_vars:
            idata.add_groups({'log_likelihood': {'logl': idata.posterior['logl_pointwise']}})
        else:
            idata = pm.compute_log_likelihood(idata)

    return idata 

# Model Comparison 
def compare_models_loo(idata_single, idata_binary):
    """Compare models using Leave-One-Out Cross-Validation"""

     # Compare using arviz.compare
    print(f"\nDETAILED COMPARISON (using az.compare):")
    comparison_dict = {
        "Single_Star": idata_single,
        "Planet_Star": idata_binary
    }
    
    comp_df = az.compare(comparison_dict)
    print(comp_df)
        
    # Compute LOO for both models
    loo_single = az.loo(idata_single)
    loo_binary = az.loo(idata_binary)
    
    # Direct comparison
    print("\n" + "-"*60)
    print("MODEL COMPARISON SUMMARY")
    print("-"*60)
    
    elpd_single = loo_single.elpd_loo
    se_single = loo_single.se
    
    elpd_binary = loo_binary.elpd_loo
    se_binary = loo_binary.se
    
    elpd_diff = elpd_binary - elpd_single
    dse = np.sqrt(comp_df["dse"].iloc[0]**2 + comp_df["dse"].iloc[1]**2)
    
    print(f"Single Star Model ELPD: {elpd_single:.2f} ± {se_single:.2f}")
    print(f"Planet-Star Model ELPD: {elpd_binary:.2f} ± {se_binary:.2f}")
    print(f"Difference (Binary - Single): {elpd_diff:.2f} ± {dse:.2f}")

    SNR = elpd_diff/dse
    print(f"LOO SNR: {SNR}")
    
    return loo_single, loo_binary, comp_df


def run_pipeline(output_dir, jobID, sourceID, t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, mstar, in_notebook=False):
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
    single_model = build_single_star_model(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned)
    binary_model = build_planet_star_model_campbell_TI(t_binned, w_binned, sig_w_binned, pf_binned, psi_binned, angles='uniform', unit_disk=False)

    start_map = time.time()

    single_initvals, binary_initvals = utils.generate_initvals_from_p0(single_model, binary_model, sourceID, output_dir, mstar)
    print(binary_initvals)
    # single_initvals, binary_initvals, metadata = generate_initvals_from_grid_search(single_model, binary_model, t_ast_yr, psi, plx_factor, ast_obs, ast_err, t.meta['MSTAR'], cores=4)
    # generate_initvals(single_model, binary_model, t_ast_yr, psi, plx_factor, ast_obs, ast_err, mstar, c_funcs)
    map_time = time.time() - start_map
    print(f"MAP optimization took: {map_time:.2f} seconds")


    # Fit models and get inference data
    if in_notebook:
        print("\nFitting Single Star Model...")
        idata_single = run_mcmc(single_model, initvals=single_initvals, init='adapt_full', progressbar=True, random_seed=42)
        print("\nFitting Planet-Star Model...")
        idata_binary = run_mcmc(binary_model, initvals=binary_initvals, target_accept=0.9, tune=2000, draws=2000, cores=4, chains=4, init='adapt_full', random_seed=42, progressbar=True)
    else:
        print("\nFitting Single Star Model...")
        idata_single = run_mcmc(single_model, initvals=single_initvals, init='adapt_full', progressbar=False, callback=make_progress_callback(1000, 1000, 4), random_seed=42)
        print("\nFitting Planet-Star Model...")
        idata_binary = run_mcmc(binary_model, initvals=binary_initvals, target_accept=0.9, tune=2000, draws=2000, cores=4, chains=4, init='adapt_full', random_seed=42, progressbar=False, callback=make_progress_callback(2000, 2000, 4))
    az.to_netcdf(idata_binary, "%s/idata/binary_%s.nc"%(save_dir, jobID))
    az.to_netcdf(idata_single, "%s/idata/single_%s.nc"%(save_dir, jobID))

    # Compare models using LOO
    loo_single, loo_binary, comp_df = compare_models_loo(idata_single, idata_binary)

    # Plot comparison
    plotting.plot_model_fits(t_binned, w_binned, sig_w_binned, idata_single, idata_binary, save_fn = save_dir / f"plots/model_comparison_fits_{jobID}.png")

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

def process_idata(idata_binary, mstar):
    processed_idata = idata_binary.copy() 
    avail_vars = [var for var in idata_binary.posterior.data_vars]

    processed_idata.posterior["delta_ra"] = processed_idata.posterior["delta_ra"] * 1e3
    processed_idata.posterior["delta_dec"] = processed_idata.posterior["delta_dec"] * 1e3
    processed_idata.posterior["pm_ra"] = processed_idata.posterior["pm_ra"] * 1e3
    processed_idata.posterior["pm_dec"] = processed_idata.posterior["pm_dec"] * 1e3

    if "mp" not in avail_vars:
        a0_AU = processed_idata.posterior['a0'].values / processed_idata.posterior['parallax'].values
        if 'log_a0' not in avail_vars:
             processed_idata.posterior['log_a0'] = (("chain", "draw"), np.log(processed_idata.posterior['a0'].values))
        mp = utils.solve_planet_mass(mstar, processed_idata.posterior['p'].values, a0_AU)
        processed_idata.posterior['mp'] = (("chain", "draw"), mp * 1047.57)
        processed_idata.posterior['log_mp'] = (("chain", "draw"), np.log(mp * 1047.57))

        sma = (mstar + mp)**(1/3) * (processed_idata.posterior['p'].values/365.25)**(2/3)
        processed_idata.posterior['sma'] = (("chain", "draw"), sma)

    if "omega" not in avail_vars:
        a0_mas, Omega, omega, incl = utils.thiele_innes_to_campbell(processed_idata.posterior["A_in_mas"].values, processed_idata.posterior["B_in_mas"].values, processed_idata.posterior["F_in_mas"].values, processed_idata.posterior["G_in_mas"].values) 
        processed_idata.posterior["Omega"] = (("chain", "draw"), Omega)
        processed_idata.posterior["omega"] = (("chain", "draw"), omega)
        processed_idata.posterior["incl"] = (("chain", "draw"), incl)
        processed_idata.posterior["cosi"] = (("chain", "draw"), np.cos(incl))

    processed_idata.posterior["a0"] = processed_idata.posterior["a0"] * 1e3
    processed_idata.posterior["parallax"] = processed_idata.posterior["parallax"] * 1e3

    return processed_idata