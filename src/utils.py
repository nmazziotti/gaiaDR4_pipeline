import numpy as np 
import scipy 
import arviz as az
import pandas as pd 
from pathlib import Path
from astropy.table import Table

user_dir = Path('/work/hdd/bfoc/nm78/')

# Helper functions
def solve_planet_mass(mstar, period, a0_AU):
    '''
    Computes the mass of a planetary companion from the resulting gaiamock fit parameters.
    Solves for planet mass by determining root with scipy.optimize.brentq, which has better
    numerical stability for low companion masses than Newton's method.

    mstar: Mass of host star (Msun)
    period: Orbital period fit (days), scalar or array-like
    a0_AU: Photocenter semimajor axis fit in AU, scalar or array-like

    Returns: planet mass in Msun (scalar in -> scalar-like out, array in -> array out)
    '''
    def _solve(period, a0_AU):
        G = 2.959e-4  # AU^3 Msun^-1 day^-2
        C = (G * period**2) / (4 * np.pi**2 * a0_AU**3)
        f = lambda mp: (mp + mstar)**2 - C * mp**3
        try:
            return scipy.optimize.brentq(f, 1e-10, 1)
        except ValueError:
            return 0.0

    return np.vectorize(_solve)(period, a0_AU)

def check_ruwe(t_ast_yr, psi, plx_factor, ast_obs, ast_err, binned = True):
    '''
    This function takes a set of astrometric data (t_ast_yr, psi, plx_factor, ast_obs, ast_err) and fits a 5-parameter solution. It inflates the uncertainties according to the goodness of fit and returns the 5-parameter UWE, best-fit parameters, and uncertainties. 
    When calculating ruwe and parallax uncertainty inflation factors, we need to account for the fact that we binned
        (averaging 8 ccds per FOV transit), because binning does not conserve reduced chi^2. 
    '''
    Cinv = np.diag(1/ast_err**2)    
    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T 
    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs)  
    Lambda_pred = np.dot(M, mu)
    resids = ast_obs - Lambda_pred
    Nobs, nu, nu_unbinned = len(ast_obs), len(ast_obs) - 5, len(ast_obs)*8 - 5  
    chi2_red_binned = np.sum(resids**2/ast_err**2)/nu

    def predict_reduced_chi2_unbinned_data(chi2_red_binned, n_param, N_points, Nbin=8):
        '''
        this function corrects for the fact that reduced chi2 for a poor fit increases when the data is binned. 
        chi2_red_binned: the reduced chi2, i.e. chi^2/(N_data - n_param), calculated from the binned data
        n_param: the number of free parameters in the model
        N_points: the number of points after binning. The number before binning is N_points*Nbin
        Nbin: how many observations are combined to make one data point. For our purposes, the number of CCDs 
        '''
        return (N_points*Nbin - N_points + chi2_red_binned*(N_points - n_param) )/(N_points*Nbin - n_param)
    
    chi2_red_unbinned = predict_reduced_chi2_unbinned_data(chi2_red_binned = chi2_red_binned, n_param = 5, N_points = Nobs, Nbin=8)
    
    if binned:
        ruwe = np.sqrt(chi2_red_unbinned)
        cc = np.sqrt(chi2_red_unbinned/((1-2/(9*nu_unbinned))**3 ))
    else:
        ruwe = np.sqrt(chi2_red_binned)
        cc = np.sqrt(chi2_red_binned/((1-2/(9*nu))**3 ))
        
    cov_matrix = np.linalg.inv(M.T @ Cinv @ M)
    sigma_mu = cc*np.sqrt(np.diag(cov_matrix))
    
    return ruwe, mu, sigma_mu
 
def P_from_sma(sma, mstar, mplanet):
    return sma**(3/2) / np.sqrt(mstar + mplanet) * 365.25

def sma_from_P(P, mstar, mplanet):
    return (mstar + mplanet)**(1/3) * (P/365.25)**(2/3)

def campbell_to_thiele_innes(a0, Omega, omega, incl):
    A = a0 * (np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(incl))
    B = a0 * (np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(incl))
    F = -a0 * (np.sin(omega) * np.cos(Omega) + np.cos(omega) * np.sin(Omega) * np.cos(incl))
    G = -a0 * (np.sin(omega) * np.sin(Omega) - np.cos(omega) * np.cos(Omega) * np.cos(incl))
    return A, B, F, G


def thiele_innes_to_campbell(A, B, F, G):
    '''
    Translate between Campbell elements and Thiele-Innes coefficients. Equations from the appendix of Halbwachs+2023. 
    Equations for uncertainties can also be found there but are more complicated and not implemented here. 
    A, B, F, G are Thiele-Innes elements in mas, provided as scalars or arrays.
    Adapted from NSSTools 
    '''
    # Compute wp - Omega and wm - Omega
    wp_minus_Omega = np.arctan2(B - F, A + G)  # Argument of periapsis + ascending node
    wm_minus_Omega = np.arctan2(-B - F, A - G)  # Argument of periapsis - ascending node

    # Initial estimates for w and Omega
    w = (wp_minus_Omega + wm_minus_Omega) / 2.0  # Argument of periapsis
    Omega = (wp_minus_Omega - wm_minus_Omega) / 2.0  # Longitude of ascending node

    # Ensure Omega is between 0 and pi
    w = np.where(Omega < 0, w + np.pi, w)  # Adjust w accordingly
    Omega = np.where(Omega < 0, Omega + np.pi, Omega)  # Adjust Omega by adding pi

    # Calculate tan^2(i/2) using two formulas
    tan2_i_AG = np.abs((A + G) * np.cos(wm_minus_Omega))
    tan2_i_BF = np.abs((F - B) * np.sin(wm_minus_Omega))

    # Choose the formula with the larger denominator for stability
    use_tan2_i_AG = tan2_i_AG > tan2_i_BF
    inclination = np.where(
        use_tan2_i_AG,
        2.0 * np.arctan2(np.sqrt(np.abs((A - G) * np.cos(wp_minus_Omega))), np.sqrt(tan2_i_AG)),
        2.0 * np.arctan2(np.sqrt(np.abs((B + F) * np.sin(wp_minus_Omega))), np.sqrt(tan2_i_BF))
    )

    # Compute semi-major axis
    u = (A**2 + B**2 + F**2 + G**2) / 2.0
    v = A * G - B * F
    sqrt_u2_minus_v2 = np.sqrt((u + v) * (u - v))
    a0 = np.sqrt(u + sqrt_u2_minus_v2)

    # Ensure w is between 0 and 2*pi
    w = np.where(w > 2 * np.pi, w - 2 * np.pi, w)
    w = np.where(w < 0, w + 2 * np.pi, w)

    # Convert to scalars if inputs are scalars
    if np.isscalar(A) and np.isscalar(B) and np.isscalar(F) and np.isscalar(G):
        return float(a0), float(Omega), float(w), float(inclination)
    return a0, Omega, w, inclination

def mod_angle(angle, range=[-np.pi,np.pi]):
    if range == [-np.pi,np.pi]:
        return (angle + np.pi) % (2*np.pi) - np.pi
    elif range == [0, 2*np.pi]:
         return angle % (2*np.pi) 

# Loading files 
def load_astrometry(sourceID, version='dr4'):
    data_dir = user_dir / "gaiamock_data"
    filename = f'EpochAstrometry-Gaia_{version.upper()}_{sourceID}.fits'
    t = Table.read(data_dir / filename)
    return t 

def load_p0(out_dir, sourceID):
    p0 = np.loadtxt(user_dir / f'/{out_dir}/runs/{sourceID}/optimize/grid_search_p0.txt')
    return p0

def load_grid_search_res(out_dir, sourceID):
    data = np.load(user_dir / f'/{out_dir}/runs/{sourceID}/optimize/grid_search.npz')
    return data 

def load_idata(out_dir, sourceID, jobID, binary=True):
    if binary:
        idata = az.from_netcdf(user_dir / f'/{out_dir}/runs/{sourceID}/idata/binary_{jobID}.nc')
    else:
        idata = az.from_netcdf(user_dir / f'/{out_dir}/runs/{sourceID}/idata/single_{jobID}.nc')
    return idata

def load_comparison_res(out_dir, sourceID, jobID):
    df_res = pd.read_csv(user_dir / f'/{out_dir}/runs/{sourceID}/csv/model_comparison_results_{jobID}.csv')
    return df_res 

def compute_loo_SNR(df_res):
    if df_res['Unnamed: 0'][0] == 'Planet_Star':
        binary_index = 0
        single_index = 1
    else:
        binary_index = 1
        single_index = 0
    
    SNR = (df_res['elpd_loo'][binary_index] - df_res['elpd_loo'][single_index]) / np.sqrt(df_res['dse'][0]**2 + df_res['dse'][1]**2)
    SNR = float(SNR)
    return SNR 

# Optimization functions  
def generate_metadata(ra_off, dec_off, pmra, pmdec, plx, mstar, **kwargs):
    binary = {"period", "ecc", "phi"}
    campbell = {"a0", "Omega", "omega", "incl"}
    thiele_innes = {"A", "B", "F", "G"}
    provided = set(kwargs.keys())

    if binary.issubset(provided):
        period, ecc, phi = kwargs["period"], kwargs["ecc"], kwargs["phi"]
        if campbell.issubset(provided):
            a0, Omega, omega, incl = kwargs["a0"], kwargs["Omega"], kwargs["omega"], kwargs["incl"]
            A, B, F, G = campbell_to_thiele_innes(a0, Omega, omega, incl)
        elif thiele_innes.issubset(provided):
            A, B, F, G = kwargs["A"], kwargs["B"], kwargs["F"], kwargs["G"]
            a0, Omega, omega, incl = thiele_innes_to_campbell(A, B, F, G)

        plus = (Omega + omega)/2
        minus = (Omega - omega)/2

        if "mp" not in kwargs:
            mp = solve_planet_mass(mstar, period, a0/plx)
        sma = (mstar + mp)**(1/3) * (period/365.25)**(2/3)

        h = ecc * np.sin(omega)
        k = ecc * np.cos(omega)

        if h == 0 and k == 0:
            h = 0.01 
            k = 0.01

        __k_unit_disk = k / np.sqrt(1 - h**2)
    
        metadata = {
            'delta_ra': np.array(ra_off * 1e-3), # as
            'delta_dec': np.array(dec_off * 1e-3), # as 
            'pm_ra': np.array(pmra * 1e-3), # as/yr
            'pm_dec': np.array(pmdec * 1e-3), # as /yr
            'log_parallax': np.array(np.log(plx*1e-3)), # as 
            'parallax': np.array(plx*1e-3), # as
            'log_p': np.array(np.log(period)), # days
            'p': np.array(period), # days
            'sma': np.array(sma), # AU
            'cosi': np.array(np.cos(incl)), 
            'ecc': np.array(ecc),
            'h': np.array(h),
            'k': np.array(k),
            '__k_unit_disk': np.array(__k_unit_disk),
            'log_mp': np.array(np.log(mp)), # Msun 
            'mp': np.array(mp), # Msun
            'plus': np.array( mod_angle(plus) ), 
            'minus': np.array( mod_angle(minus) ), 
            'Omega': np.array( mod_angle(Omega, range=[0, 2*np.pi]) ),
            'omega': np.array( mod_angle(omega, range=[0, 2*np.pi]) ),
            'phi': np.array(mod_angle(phi)), 
            'A_in_mas': np.array(A),
            'B_in_mas': np.array(B),
            'F_in_mas': np.array(F),
            'G_in_mas': np.array(G),
            'a0': np.array(a0 * 1e-3), # as
            'log_a0': np.array(np.log(a0 * 1e-3)), # as
            }
    else:
        metadata = {
            'delta_ra': np.array(ra_off * 1e-3), # as
            'delta_dec': np.array(dec_off * 1e-3), # as 
            'pm_ra': np.array(pmra * 1e-3), # as/yr
            'pm_dec': np.array(pmdec * 1e-3), # as /yr
            'log_parallax': np.array(np.log(plx*1e-3)), # as 
            'parallax': np.array(plx*1e-3) # as
            }
    return metadata

def initvals_for_model(model, metadata):
    def compute_pymc_interval(x, lower, upper):
        return np.array(np.log((x - lower) / (upper - x)))

    initvals = {}
    for var in model.basic_RVs:
        name = var.name
        dist = type(var.owner.op).__name__  # "Uniform"

        if name != 'logl':
            if 'Uniform' in dist:
                lower = var.owner.inputs[2].eval()
                upper = var.owner.inputs[3].eval()

                guess =  metadata[name]
                if guess <= lower:
                    guess = lower + 1e-6
                elif guess >= upper :
                    guess = upper - 1e-6

                initvals[name + "_interval__"] = compute_pymc_interval(guess, lower, upper)
            elif "angle1" in name:
                angle_name = name.split('_')[2]

                initvals[name] = np.array(np.sin(metadata[angle_name]))
            elif "angle2" in name:
                angle_name = name.split('_')[2]

                initvals[name] = np.array(np.cos(metadata[angle_name]))
            else:
                initvals[name] = metadata[name]
    return initvals

def convert_p0_to_metadata(out_dir, sourceID, mstar):
    p0 = load_p0(out_dir, sourceID)
    log_p, ecc, phi, ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = p0
    a0, Omega, omega, incl = thiele_innes_to_campbell(A, B, F, G)
    period = np.exp(log_p)
    metadata = generate_metadata(ra_off, dec_off, pmra, pmdec, plx, mstar, period, ecc, phi, a0, Omega, omega, incl)
    return metadata 

def generate_initvals_from_p0(single_model, binary_model, sourceID, out_dir, mstar):
    metadata =  convert_p0_to_metadata(out_dir, sourceID, mstar)
    
    single_initvals = initvals_for_model(single_model, metadata)
    binary_initvals = initvals_for_model(binary_model, metadata)

    return single_initvals, binary_initvals

