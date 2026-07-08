import numpy as np 
import scipy 
import arviz as az
import pandas as pd 
from pathlib import Path
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner

from . import utils, mcmc

def plot_grid_search_chi2(cwd, sourceID, ax=None):
    res = utils.load_grid_search_res(cwd, sourceID)
    chi2min_logP = res["log_P"][np.argmin(res["min_chi2"])]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(res["log_P"], res["min_chi2"], marker='o', markersize=2, markerfacecolor='k', markeredgecolor='none')
    ax.set_xlabel(r"$\log{P}$ [days]")
    ax.set_ylabel(r"Binary Model Marginalized $\chi^2$")
    ax.axvline(chi2min_logP, label=r"Global Min $\chi^2$" + f"\n({np.exp(chi2min_logP):.2f} days)", linestyle=":", c='gray')
    ax.legend()

    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())  # match the range exactly
    top_ax.set_xlabel(r'$P$ [days]')

    bottom_ticks = ax.get_xticks()
    xmin, xmax = ax.get_xlim()
    bottom_ticks = [t for t in bottom_ticks if xmin <= t <= xmax]
    top_ax.set_xticks(bottom_ticks)  # same positions as bottom axis
    top_ax.set_xticklabels([f'{np.exp(t):.0f}' for t in bottom_ticks])


def arviz_summary(enhanced_idata):
        # --- Add a table (top half) ---
        idata = enhanced_idata.copy()

        idata.posterior["delta_ra [mas]"] = idata.posterior["delta_ra"] 
        idata.posterior["delta_dec [mas]"] = idata.posterior["delta_dec"] 
        idata.posterior["delta_ra [mas]"] = idata.posterior["delta_ra"]
        idata.posterior["pm_ra [mas/yr]"] = idata.posterior["pm_ra"] 
        idata.posterior["pm_dec [mas/yr]"] = idata.posterior["pm_dec"] 
        idata.posterior["parallax [mas]"] = idata.posterior["parallax"] 
        idata.posterior["mp [mjup]"] = idata.posterior["mp"]
        idata.posterior["p [days]"] = idata.posterior["p"]
        idata.posterior["a0 [mas]"] = idata.posterior["a0"]
        idata.posterior["sma [AU]"] = idata.posterior["sma"]

        summary = az.summary(idata, round_to="none", var_names=["delta_ra [mas]", "delta_dec [mas]", "pm_ra [mas/yr]", "pm_dec [mas/yr]", "parallax [mas]", 
                                                                "p [days]", "sma [AU]", "ecc", "phi", "a0 [mas]", "Omega", "omega", "cosi", "mp [mjup]"])
        df = summary

        # if t.meta['PERIOD'] != 0.0:
        #     true_phi = 2*np.pi*t.meta['TP']/t.meta['PERIOD']
        # else:
        #     true_phi = 0.0

        # df["truth"] = [None, None, t.meta["PMRA"], t.meta["PMDEC"], 1e3/t.meta["DIST"], 
        #             t.meta["MPLANET"], t.meta["PERIOD"], t.meta["SMA"], t.meta["ECC"], true_phi, 
        #             t.meta["OMEGA"], t.meta["W"], np.cos(t.meta["INCL"])]
        # cols = ["truth"] + df.columns.drop("truth").tolist()
        # df = df.reindex(columns=cols)

        for column in df.columns:
            if column == 'r_hat' or 'mcse' in column:
                df[column] = df[column].round(4) 
            elif column == 'sd':
                df[column] = df[column].round(3) 
            else:
                df[column] = df[column].round(2) 
        return df

def generate_corner_plot(idata, model='binary', fig=None, use_campbell=True, true_vals=None, initvals=None):
        posterior = idata.posterior 

        if model == 'binary': 
            if use_campbell:
                plot_list = ['delta_ra', 'delta_dec', 'pm_ra', 'pm_dec', 'parallax', 'p', 'ecc', 'phi', 'a0', 'Omega', 'omega',  'cosi', 'mp']
                labels = [
                            r"$\Delta\alpha$ [mas]", r"$\Delta\delta$ [mas]", r"$\mu_{\alpha}$ [mas/yr]", r"$\mu_{\delta}$ [mas/yr]",
                            r"$\varpi$ [mas]", r"${P}$ [days]", r"$e$", r"$\phi$", r"$a_0$ [mas]", 
                            r"$\Omega$", r"$\omega$", r"$\cos{i}$", r"$M_P$ [$M_J$]"
                        ]
            else:
                plot_list = ['delta_ra', 'delta_dec', 'pm_ra', 'pm_dec', 'parallax', 'p', 'ecc', 'phi', 'A_in_mas', 'B_in_mas', 'F_in_mas', 'G_in_mas', 'mp']
                labels = [
                            r"$\Delta\alpha$ [mas]", r"$\Delta\delta$ [mas]", r"$\mu_{\alpha}$ [mas/yr]", r"$\mu_{\delta}$ [mas/yr]",
                            r"$\varpi$ [mas]", r"${P}$ [days]", r"$e$", r"$\phi$",
                            r"A [mas]", r"B [mas]", r"F [mas]", r"G [mas]", r"$M_P$ [$M_J$]"
                        ]
                
        elif model == 'single':
            plot_list = ['delta_ra', 'delta_dec', 'pm_ra', 'pm_dec', 'parallax']
            labels = [r"$\Delta\alpha$ [mas]", r"$\Delta\delta$ [mas]", r"$\mu_{\alpha}$ [mas/yr]", r"$\mu_{\delta}$ [mas/yr]", r"$\varpi$ [as]"]

        corner_kwargs = dict(
                        labels=labels,
                        truth_color='red',
                        show_titles=True,
                        plot_datapoints=False,
                        title_kwargs={"fontsize": 8}
                    )
        if fig is not None:
            corner_kwargs['fig'] = fig
        if true_vals is not None:
                truth_list = [true_vals[name] for name in plot_list]
                corner_kwargs['truths'] = truth_list 
        corner_kwargs['var_names'] = plot_list
        corner_fig = corner.corner(posterior, **corner_kwargs)
        
        if initvals is not None:
            initval_list = []
            for name in plot_list:
                if name in initvals.keys():
                    if name == 'pm_ra' or name =='pm_dec' or name == 'delta_dec' or name == 'delta_ra' or name == 'parallax' or name == 'a0':
                        initval_list.append(initvals[name] * 1e3)
                    elif name == 'mp':
                        initval_list.append(initvals[name] * 1047.57)
                    else:
                        initval_list.append(initvals[name])
                else:
                    initval_list.append(None)
            axes = np.array(corner_fig.axes).reshape(len(plot_list), len(plot_list))
            for i, val in enumerate(initval_list):
                if val is not None:
                    axes[i, i].axvline(val, color='k', lw=1.5, linestyle=':')            
        return corner_fig

def plot_model_fits(t_binned, w_binned, sig_w_binned, idata_single, idata_binary, save_fn=None, axes=None):
    """Plot model fits and residuals"""
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Single star model
    ax = axes[0, 0]
    ax.errorbar(t_binned, w_binned, yerr=sig_w_binned, fmt='o', ms=3, c='k', alpha=0.7, label='Data')
    
    as_pred_single = idata_single.posterior['model_single'].values
    q16, q50, q84 = np.percentile(as_pred_single, [16, 50, 84], axis=(0, 1))
    
    ax.plot(t_binned, q50, lw=2, color='blue', label='Single Star Model')
    ax.fill_between(t_binned, q16, q84, alpha=0.3, color='blue')
    ax.set_ylabel('Along-Scan [arcsec]')
    ax.set_title('Single Star Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Single star residuals
    ax = axes[1, 0]
    ax.errorbar(t_binned, w_binned - q50, yerr=sig_w_binned, fmt='o', ms=3, c='k', alpha=0.7)
    ax.axhline(0, color='blue', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time [BJD]')
    ax.set_ylabel('Residual [arcsec]')
    ax.set_title('Single Star Residuals')
    ax.grid(True, alpha=0.3)
    
    # Planet-star model
    ax = axes[0, 1]
    ax.errorbar(t_binned, w_binned, yerr=sig_w_binned, fmt='o', ms=3, c='k', alpha=0.7, label='Data')
    
    as_pred_binary = idata_binary.posterior['model_single'].values + idata_binary.posterior['model_binary'].values
    q16, q50, q84 = np.percentile(as_pred_binary, [16, 50, 84], axis=(0, 1))
    
    ax.plot(t_binned, q50, lw=2, color='red', label='Planet-Star Model')
    ax.fill_between(t_binned, q16, q84, alpha=0.3, color='red')
    ax.set_ylabel('Along-Scan [arcsec]')
    ax.set_title('Planet-Star Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(labelleft=False, labelright=True)
    
    # Planet-star residuals
    ax = axes[1, 1]
    ax.errorbar(t_binned, w_binned - q50, yerr=sig_w_binned, fmt='o', ms=3, c='k', alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time [BJD]')
    ax.set_ylabel('Residual [arcsec]')
    ax.set_title('Planet-Star Residuals')
    ax.grid(True, alpha=0.3)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(labelleft=False, labelright=True)
    
    #plt.tight_layout()
    if save_fn is not None:
        plt.savefig(save_fn, dpi=300, bbox_inches='tight')
        plt.close()

def kepler(M, ecc, tol=1e-10, max_iter=100):
    E = M.copy()
    for _ in range(max_iter):
        dE = (M - E + ecc*np.sin(E)) / (1 - ecc*np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    f = 2 * np.arctan2(
        np.sqrt(1+ecc) * np.sin(E/2),
        np.sqrt(1-ecc) * np.cos(E/2)
    )
    return f

def get_position(t_binned, p, ecc, phi, A, B, F, G):
    # Orbital motion
    n = 2*np.pi / p
    M = n * t_binned - phi
    #f = xo.orbits.keplerian.get_true_anomaly(M, ecc + pt.zeros_like(M))
    f = kepler(M, ecc)
    r = (1 - ecc**2) / (1 + ecc*np.cos(f))
    X = r * np.cos(f)
    Y = r * np.sin(f)

    b_ra = A*X + F*Y
    b_dec = B*X + G*Y
    return b_ra, b_dec

def plot_posterior_orbits(enhanced_idata, t_binned, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6), dpi=600)


        p_posterior = enhanced_idata.posterior['p'].values.flatten()
        ecc_posterior = enhanced_idata.posterior['ecc'].values.flatten()
        phi_posterior = enhanced_idata.posterior['phi'].values.flatten()

        A_posterior = enhanced_idata.posterior['A_in_mas'].values.flatten()
        B_posterior = enhanced_idata.posterior['B_in_mas'].values.flatten()
        F_posterior = enhanced_idata.posterior['F_in_mas'].values.flatten()
        G_posterior = enhanced_idata.posterior['G_in_mas'].values.flatten()

        draws = np.random.randint(low=0, high=len(p_posterior), size=300)

        n_pts = len(t_binned)
        cmap = plt.get_cmap('seismic', n_pts)
        colors = cmap(np.arange(n_pts))

        for j, index in enumerate(draws):
            p = p_posterior[index]
            ecc = ecc_posterior[index]
            phi = phi_posterior[index]
            A = A_posterior[index]
            B = B_posterior[index]
            F = F_posterior[index]
            G = G_posterior[index]

            b_ra, b_dec = get_position(np.linspace(0, 1.1 * p, 1000), p, ecc, phi, A, B, F, G)
            b_ra_data, b_dec_data = get_position(t_binned, p, ecc, phi, A, B, F, G)

            ax.plot(b_ra, b_dec, lw=0.5, color='tab:blue', alpha=0.3, zorder=1)
            ax.scatter(b_ra_data, b_dec_data, lw=0.5, alpha=0.3, s=2, zorder=10, c=colors)

        ax.plot(0, 0, marker='x', c='k')
        ax.set_xlabel(r'$\Delta\alpha\cos\delta$ [mas]')
        ax.set_ylabel(r'$\Delta\delta$ [mas]')

        ax.invert_xaxis()
        ax.set_aspect('equal')
        ax.set_title("Posterior-Sampled Sky-Plane Orbit")

        # colorbar
        norm = plt.Normalize(vmin=t_binned.min(), vmax=t_binned.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = ax.figure.colorbar(sm, cax=cax)
        #, fraction=0.046, pad=0.04
        cbar.set_label('Time [BJD]') 


def summary_plot(cwd, sourceID, jobID, mstar, ra, dec, t_binned, w_binned, sig_w_binned):
    idata_single = utils.load_idata(cwd, sourceID, jobID, binary=False)
    idata_binary = utils.load_idata(cwd, sourceID, jobID)
    sampling_runtime = idata_binary.sample_stats.sampling_time
    df_res = utils.load_comparison_res(cwd, sourceID, jobID)
    #df_timing = pd.read_csv(cwd / f'mcmc_files/{sourceID}/csv/timing_{jobID}.csv')
    SNR_detection, loo_binary, loo_single = utils.compute_loo_SNR(df_res)

    processed_idata = mcmc.process_idata(idata_binary, mstar)

    df = arviz_summary(processed_idata)
    #compute_SNR(df_res)
    #sampling_runtime = df_timing["Binary_Sampling"][0]

    fig = plt.figure(figsize=(30, 22.5), dpi=150)

    # Outer grid: 2 rows, 2 columns
    gs_outer = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.5, 1],
                                height_ratios=[1, 2.5],  # top row smaller, bottom row taller
                                left=0.05, right=0.97, top=0.97, bottom=0.03,
                                wspace=0.1, hspace=0.1)

    # Table: top left
    ax_table = fig.add_subplot(gs_outer[0, 0])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        bbox=[0.1, 0.1, 1.0, 1.0]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 2.2)

    subfig_corner = fig.add_subfigure(gs_outer[1, 0])
    initvals = utils.convert_p0_to_metadata(cwd, sourceID, mstar)
    corner_fig = generate_corner_plot(processed_idata, initvals=initvals, use_campbell=True, fig=subfig_corner);
    #plot_corner(enhanced_idata, sourceID, fig=subfig_corner, initvals=initvals)

    for ax in subfig_corner.get_axes():
        ax.tick_params(axis='both', labelsize=5)
        for col in ax.collections:
            col.set_linewidth(0.5)
        ax.xaxis.offsetText.set_fontsize(5)
        ax.yaxis.offsetText.set_fontsize(5)
        # ax.set_rasterized(True)

    # Right column bottom: two stacked plots
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1, 1],
                                                height_ratios=[1, 1], hspace=0.2)


    gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_right[0], hspace=0.25, wspace=0.1)
    
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
    ax10 = fig.add_subplot(gs[1, 0], sharex=ax00)
    ax11 = fig.add_subplot(gs[1, 1], sharex=ax01)
    
    axes = np.array([[ax00, ax01], [ax10, ax11]])
    plot_model_fits(t_binned, w_binned, sig_w_binned, idata_single, idata_binary, save_fn=None, axes=axes)

    ax_orbit = fig.add_subplot(gs_right[1])
    plot_posterior_orbits(processed_idata, t_binned, ax=ax_orbit)

    fig.text(0.83, 0.97, f"SRCID: {sourceID}",
            ha="center", va="top",
            fontsize=25, fontweight="bold")

    df_stats = pd.DataFrame.from_dict({
        "Mstar [Msun]": mstar,
        "RA": np.round(ra, 2),
        "Dec": np.round(dec, 2),
        "Single-star LOO": np.round(loo_single,3), 
        "Planet-star LOO": np.round(loo_binary,3), 
        "LOO SNR": np.round(SNR_detection,3), "Runtime [min]": np.round(sampling_runtime/60, 3)}, orient="index", columns=["Value"])

    ax_table = fig.add_subplot(gs_outer[0, 1])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=df_stats.values,
        rowLabels=df_stats.index,
        colLabels=df_stats.columns,
        loc="center",
        cellLoc="center",
        bbox=[0.6, 0.1, 0.2, 0.7]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(20)


    n_axes = len(subfig_corner.axes)
    ndim = int(np.sqrt(n_axes))
    corner_axes = np.array(subfig_corner.axes).reshape(ndim, ndim)


    r0, r1 = 0, 4
    c0, c1 = 7, ndim - 1


    gs_inner = corner_axes[r0, c0].get_subplotspec().get_gridspec()


    for i in range(r0, r1):
        for j in range(c0, c1):
            corner_axes[i, j].remove()


    ax_extra = subfig_corner.add_subplot(gs_inner[r0:r1, c0:c1])
    plot_grid_search_chi2(cwd, sourceID, ax=ax_extra)

    fig.canvas.draw()
    return fig