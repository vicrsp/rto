from .data_model import RTODataModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import isnan

class ExperimentAnalyzer:
    def __init__(self, db_file):
        self.md = RTODataModel(db_file)

    def load(self, rto_type):
       return pd.DataFrame(self.md.get_rto_experiment_results(rto_type), columns=['rto.id', 'rto.name', 'rto.type', 'run.id', 'run.status', 'iteration', 'var_name', 'value'])

    def load_by_id(self, id):
       return pd.DataFrame(self.md.get_rto_experiment_results_by_id(id), columns=['rto.id', 'rto.name', 'rto.type', 'run.id', 'run.status', 'iteration', 'var_name', 'value'])

    def pre_process(self, results, f_plant=None, u_plant=None):
        def aggfunc(x):
            return x
        # Transform the data
        results_pv = pd.pivot_table(results, values='value', index=['run.id','iteration','rto.type','run.status'], columns=['var_name'], aggfunc=aggfunc)
        results_pv.reset_index(level=results_pv.index.names, inplace=True)
        
        # Convert the values
        results_pv[['cost_model','cost_real','fobj_modifier', 'opt_time']] = results_pv[['cost_model','cost_real','fobj_modifier','opt_time']].astype('float')
        # Get the inputs
        results_pv['u'] = results_pv['u'].apply(lambda x: np.array([float(xi) for xi in x.split(',')]))
        results_pv['u_opt'] = results_pv['u_opt'].apply(lambda x: np.array([float(xi) if xi else np.NaN for xi in str(x).split(',')]))

        # Extract some variables
        results_pv['g_0'] = results_pv['g_real'].apply(lambda x: float(x.split(',')[0])) 
        results_pv['g_1'] = results_pv['g_real'].apply(lambda x: float(x.split(',')[1])) 
        results_pv['g_0_model'] = results_pv['g_model'].apply(lambda x: float(x.split(',')[0])) 
        results_pv['g_1_model'] = results_pv['g_model'].apply(lambda x: float(x.split(',')[1])) 
        results_pv['g_0_modifiers'] = results_pv['g_modifiers'].apply(lambda x: float(x.split(',')[0])) 
        results_pv['g_1_modifiers'] = results_pv['g_modifiers'].apply(lambda x: float(x.split(',')[1])) 

        # kpis
        if(u_plant is not None):
            results_pv['du'] = results_pv['u'].apply(lambda x: np.linalg.norm(100 * (x - u_plant)/u_plant))
        
        if(f_plant is not None):
            results_pv['dPhi'] = results_pv[['cost_real']].apply(lambda x: 100 * np.abs((x - f_plant)/f_plant))

        return results_pv
    
    def plot_by_iteration(self, data, y, ylabel, title='', style=None, hue='run.status', xlabel='Iteration'):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.lineplot(data=data, y=y, x='iteration', hue=hue, style=style, ax=ax, palette='Set1', seed=1234, legend=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.get_legend().set_title('')
        ax.set_title(title)
        fig.show()
        return ax, fig
    
    def load_run_models(self, run_id):
        return self.md.get_run_models(run_id)