
import  numpy as np
import pandas as pd
from scipy import stats
from typing import List,Dict,Optional,Tuple, Any


def list_columns(data_raw : Dict[str,List[Any]]) -> List[str]:
    """
    List all column names available in the raw data dict
    

   
    """
    return list(data_raw.keys())
    
def compute_empirical_distribution(samples: List[float],bins: Optional[int]=None) -> Dict[str,Any]:
    """
    Compute histogram and sample statistics (mean,variance, skewness, kurtosis
    """
    x=np.asarray(samples,dtype=float)
    n=x.size
    if bins is None:
        bins=int(max(10,np.sqrt(n)))
    counts,edges=np.histogram(x,bins=bins, density=False)
    stats_dict={
        "n":n,
        "mean":float(np.mean(x)),
        "variance": float(np.var(x,ddof=1)),
        "skewness": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
            
            
        }
    return {
        "histogram":{"bin_edges": edges.tolist(),"counts": counts.tolist()},
        "sample_stats": stats_dict
        
        
    }

def fit_standard_distribution(samples: List[float], families :List[str]=["normal","exponential","gamma"])->List[Dict[str,Any]]:
    """
    Fit each named distribution by MLE and compute a KS goodness of fit staatistic.
    Returns list of dicts with keys: family, fitted_params, gof_stat.
    """
    x=np.asarray(samples,dtype=float)
    results=[]
    for fam in families:
        fam=fam.lower()
        if fam== "normal":
            mu,sigma=stats.norm.fit(x)
            cdf= lambda v: stats.norm.cdf(v,loc=mu,scale=sigma)
            params={"mu":float(mu),"sigma":float(sigma)}
        elif fam== "exponential":
            loc,scale=stats.expon.fit(x,floc=0)
            cdf=lambda v : stats.expon.cdf(v,loc=loc,scale=scale)
            params={"rate": float(1/scale)}
        elif fam == "gamma":
            # gamma.fit returns (a, loc, scale)
            a, loc, scale = stats.gamma.fit(x, floc=0)
            cdf = lambda v: stats.gamma.cdf(v, a, loc=loc, scale=scale)
            params = {"alpha": float(a), "beta": float(1/scale)}
        else:
            continue
        #kolmogorov-smirnov test
        ks_stat, _= stats.kstest(x,cdf)
        results.append({
            "family":fam,
            "fitted_params":params,
            "gof_stat": float(ks_stat)
        })
    return sorted(results,key=lambda r : r["gof_stat"])


def select_closest_distribution(fit_results: List[Dict[str,Any]],top_k : int =1)->Tuple[List[str],List[Dict[str,Any]]]:
    """
    Return the top_k families and their param dicts with the smallest gof stat.
    """ 
    chosen=fit_results[:top_k] 
    families=[r["family"] for r in chosen ]
    params=[r["fitted_params"] for r in chosen]
    
    return families,params


def query_parameters_to_test(
    chosen_family: str
)->List[str]:
    """
    For a given distribution family,return the list of available parameters to test.
    """
    mapping={
        "normal":["mean","variance"],
        "exponential":["rate"],
        "gamma" : ["alpha","beta"]
        }
    return mapping.get(chosen_family.lower(),[])
def collect_population_values(to_test: List[str],user_inputs: Dict[str,Optional[float]])-> Dict[str,Optional[float]]:
    """
    Given the parameters the user wishes to test and a dict of user inputs
    (strings or floats ) , convert to a dict of floats or None.
    """
    pop_params : Dict[str,Optional[float]]={}
    for param in to_test:
        val=user_inputs.get(param)
        if val is None or (isinstance(val,str) and val.lower() in ("","unknown","none")):
            pop_params[param]=None
        else:
            pop_params[param]=float(val)
    return pop_params

def estimate_sample_parameters(samples: List[float],to_test : List[str])-> Dict[str,float]:
    """
    Compute unbiased sample estimators for requested params
    """
    x=np.asarray(samples,dtype=float)
    estimates : Dict[str,float]={}
    if "mean" in to_test:
        estimates["mean"]=float(np.mean(x))
    if "variance" in to_test:
        estimates["variance"]=float(np.var(x,ddof=1))
    if "rate" in to_test:
        estimates["rate"]=float(1/np.mean(x))
    if "alpha" in to_test or "beta" in to_test:
        m=np.mean(x)
        v=np.var(x,ddof=1)
        alpha=m*m/v
        beta=v/m
        if "alpha" in to_test:
            estimates["alpha"]=float(alpha)
        if "beta" in to_test:
            estimates["beta"] = float(beta)
    return estimates
    
    
def decide_test_statistic(chosen_family : str, pop_params: Dict[str,Optional[float]])->Dict[str,Any]:
    """
    Decide which test statistic is feasible given known/unknown pop params.
    Return statistic name and requirements.
    """
    fam=chosen_family.lower()
    stat : Optional[str]=None
    reqs : Dict[str,bool]={}
    if fam == "normal":
        mu0, var0 = pop_params.get("mean"), pop_params.get("variance")
        # z-test if variance known & mean tested
        if mu0 is not None and var0 is not None:
            stat = "z_test"
            reqs = {"mean_known": True, "variance_known": True}
        # t-test if mean tested but variance unknown
        elif mu0 is not None and var0 is None:
            stat = "t_test"
            reqs = {"mean_known": True, "variance_known": False}
        # chi2-var test if variance tested & mean known
        elif var0 is not None and mu0 is not None:
            stat = "chi2_var"
            reqs = {"variance_known": True, "mean_known": True}
    elif fam == "exponential":
        # test on rate: use one-sample exponential test?
        lam0 = pop_params.get("rate")
        if lam0 is not None:
            stat = "exp_rate_test"
            reqs = {"rate_known": True}
    elif fam == "gamma":
        # generally need both known to do likelihood ratio, skip for now
        pass

    return {"statistic": stat, "requirements": reqs}


def compute_test_statistic( samples :List[float], pop_params : Dict[str, float], statistic :str)-> Dict[str,Any]:
    """
    Compute observed test statistic and p-value .
    Supports "z_test",t_test","chi2_var","exp_rate_test".
    """
    x=np.asarray(samples,dtype=float)
    n=x.size
    mean=np.mean(x)
    var=np.var(x,ddof=1)
    if statistic=="z_test":
        mu0=pop_params["mean"]
        sigma0=np.sqrt(pop_params["variance"])
        T=(mean-mu0)/(sigma0/np.sqrt(n))
        p = 2 * (1 - stats.norm.cdf(abs(T)))
        return {"T_obs": float(T), "p_value": float(p), "df": None, "dist": "N(0,1)"}
    elif statistic=="t_test":
        mu0=pop_params["mean"]
        s=np.sqrt(var)
        T=(mean-mu0)/(s/np.sqrt(n))
        df=n-1
        p=2*(1-stats.t.cdf(abs(T),df=df))
        return {"T_obs": float(T), "p_value": float(p), "df": df, "dist": f"t({df})"}
    
    elif statistic=="chi2_var":
        var0 = pop_params["variance"]
        T = (n - 1) * var / var0
        df = n - 1
        p = 1 - stats.chi2.cdf(T, df=df)
        return {"T_obs": float(T), "p_value": float(p), "df": df, "dist": f"chi2({df})"}
    else:
        raise ValueError(f"Unknown statistic '{statistic}'")

def interpret_results(test_output: Dict[str, Any], alpha: float = 0.05) -> str:
    """
    Return a plain-English interpretation of the test result.
    """
    T = test_output["T_obs"]
    p = test_output["p_value"]
    dist = test_output["dist"]
    decision = "reject H₀" if p < alpha else "fail to reject H₀"
    return (
        f"Test statistic {T:.3f} follows {dist}. "
        f"With p-value = {p:.4f}, at α = {alpha}, we {decision}."
    ) 

def human_input_text(prompt: str) -> str:
    """
    A simple tool that prints `prompt` and returns whatever the user types.
    The agent will call this whenever it needs to ask the user a question.
    """
    return input(prompt)


    
    