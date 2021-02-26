#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:56:43 2020

@author: gytm3
"""

import xarray as xr, pandas as pd, numpy as np, matplotlib.pyplot as plt, datetime
from sklearn import linear_model
from matplotlib.pyplot import cm
from numba import jit
from scipy import stats
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm

def QQmatch(obs,sim,pctls,sim_i,extrap=False):
     
    """        
        obs = self explanatory
        sim = hindcast
        sim_i = projection points
        pctls = numpy array of percentiles (increasing order)
        extrap = if true, linear extrapolation is performed outside obs range
        
    """

    # get percentiles of obs/sim
    obs_x=np.nanpercentile(sim,pctls) # transform from 
    obs_y=np.nanpercentile(obs,pctls) # transform to 
    simCorr=np.zeros(sim_i.shape)*np.nan

    # interpolate sim_i 
    in_bounds=np.logical_and(sim_i>=np.min(obs_x),sim_i<=np.max(obs_x))
    simCorr[in_bounds]=np.interp(sim_i[in_bounds],obs_x,obs_y) # in bounds
    
    if extrap: # linear functions
        out_bounds=np.logical_or(sim_i<np.min(obs_x),sim_i>np.max(obs_x)) 
        ps=np.polyfit(obs_x,obs_y,1) # difference in percentiles
        simCorr[out_bounds]=np.polyval(ps,sim_i[out_bounds])
        
    else: # constant correction
        above=sim_i>np.max(obs_x)
        below=sim_i<np.min(obs_x)
        deltas=obs_y-obs_x
        simCorr[above]=sim_i[above]+deltas[-1] # add diff between top percentile
        simCorr[below]=sim_i[below]+deltas[0] # add diff between bottom percentile
                    
    # return    
    return simCorr


# # Manual inspection suggests dubious data quality during this period
st=datetime.datetime(year=2019,month=12,day=27,hour=17)
stp=datetime.datetime(year=2020,month=1,day=2,hour=6)

# Should limit all data to this period
stp_gl=datetime.datetime(year=2020,month=1,day=5,hour=11)

thresh=1 # Filter out hourly-mean winds < this value

# Lead time forecast
leads=[6,12,18,24,30,36,42,48,54,60,66,72,78,84,96,102,108,114,120,126]

# Import the Forecast
dateparse = lambda x: pd.datetime.strptime(x, '%Y/%m/%d %H:%M:%S')
fr="/home/lunet/gytm3/Everest2019/Research/DangerousWinds/Data/fclog_new_South_Col.txt"
figdir="/home/lunet/gytm3/Everest2019/Research/DangerousWinds/Figures/"
r=pd.read_csv(fr,parse_dates=['init_date'], date_parser=dateparse,delimiter="\t")
r["valid_date"]=pd.to_datetime(r["valid_date"])
r["dT"]=[ii.total_seconds()/(3600) for ii in r["valid_date"]-r["init_date"]]
r.index=r["valid_date"]
r["ws"]=(r["u"]**2+r["v"]**2)**0.5

# Ditto observations -- incl. filtering and alocation
fo="/home/lunet/gytm3/Everest2019/Research/DangerousWinds/Data/south_col.csv"
o=pd.read_csv(fo,parse_dates=True,index_col=0)
for i in o.columns: o[i]=o[i].astype(np.float)
o=o.loc[o.index<=stp_gl]
u=o["WS_AVG_2"]
ug=o["WS_MAX_2"]
idx=np.logical_and(u>thresh,ug>thresh) # very minimal filter that removes wind during v. slack conditions
idx=np.logical_and(idx,~np.logical_and(u.index>=st,u.index<=stp))
u.values[~idx]=np.nan
ug.values[~idx]=np.nan


# Clip to the forecast -- only look at  the 48-hour forecast
ls=[48,]
fig,ax=plt.subplots(1,2)
count=0
xref=np.arange(30)
refline=np.linspace(0,55,100)

for l in ls:
    uf_sub=r.loc[np.logical_and(r.index.isin(ug.index),r["dT"]==l)]["ws"]
    ug_sub=ug.loc[ug.index.isin(uf_sub.index)]
    idx=np.logical_and(~np.isnan(uf_sub),~np.isnan(ug_sub))
    resid=ug_sub[idx]-uf_sub[idx]
    rbias=np.corrcoef(resid,ug_sub[idx])
    mae_raw=np.mean(np.abs(resid))
    uf_sub=uf_sub+np.mean(resid)
    ecdf_r=uf_sub[idx][np.argsort(uf_sub[idx])]
    ecdf_g=ug_sub[idx][np.argsort(ug_sub[idx])]
    x=np.arange(np.sum(idx))/np.sum(idx).astype(np.float)
    pctl=np.interp(30.,ecdf_g,x)
    y=np.linspace(0,pctl,100)
    # fig,ax=plt.subplots(1,1)
    ax.flat[0].plot(ecdf_r.values[:],x,color="grey",linewidth=2,alpha=1)
    if count ==0:
        ax.flat[0].plot(ecdf_g.values[:],x,color="black",linewidth=2)
        ax.flat[1].plot(refline,refline,color='red')
    pearson_r=uf_sub.corr(ug)
    print("l=%.0f | Pearson = %.2f"%(l,pearson_r))
    thresh=np.interp(pctl,x,ecdf_r)
    print("l=%.0f | Thresh = %.2fm/s"%(l,thresh))
    ax.flat[0].plot(xref,np.ones(len(xref))*pctl,color='k',linestyle="--")
    ax.flat[0].plot(np.ones(len(y))*30,y,color='k',linestyle="--")
    ax.flat[0].plot(np.ones(len(y))*thresh,y,color='grey',linestyle="--")
    
    corrected=QQmatch(ug_sub[idx],uf_sub[idx],np.linspace(1,100,100),
                      uf_sub[idx],extrap=False)
    mae_corr=np.mean(np.abs(corrected-ug_sub[idx]))
    print("MAE raw | corrected = %.2f | %.2f [m/s]"%(mae_raw,mae_corr))
    print("... reduced by %.0f%%"%((1-mae_corr/mae_raw)*100))
    # ax.flat[1].scatter(uf_sub[idx],ug_sub[idx],color='k',s=1)
    ax.flat[1].scatter(corrected,ug_sub[idx],color='k',s=1)
    # ax.axhline(pctl/100.,xmin=0,xmax=30.,color='k',linestyle="--")
    count+=1
    
    # ecdf_r2=corrected[np.argsort(corrected)]
    # ax.flat[0].plot(ecdf_r2,x,color='red')

    
ax.flat[0].set_ylim([0,1.01])
ax.flat[0].set_xlim([0,55])
ax.flat[0].grid(); ax.flat[1].grid()
ax.flat[0].set_xlabel("Wind Speed (m s$^{-1}$)")
ax.flat[0].set_ylabel("ECDF")
ax.flat[1].set_ylabel("Observed Wind Speed (m s$^{-1}$)")
ax.flat[1].set_xlabel("Forecast Wind Speed (m s$^{-1}$)")
ax.flat[1].set_xlim(0,55)
ax.flat[1].set_ylim(0,55)
fig.set_size_inches(9,4)
fig.savefig(figdir+"GFS_perf.tif",dpi=300)
