#!/usr/bin/env python
# coding: utf-8

import xarray as xr, pandas as pd, numpy as np, matplotlib.pyplot as plt, \
datetime
from sklearn import linear_model
from matplotlib.pyplot import cm
from numba import jit
from scipy import stats
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
dper =180./np.pi # Degrees per radian

# Circular statistics 
def circ_stat(series,wdir,span,stat,res):
    
    """
    Info to follow 
    """
    
    # Radians
    rad=np.radians(wdir)
    thresh=np.radians(span)
    circ=2*np.pi
    
    # Associate function - making sure it's 'nan-compatible'
    stat="nan"+stat
    f = getattr(np,stat)
    
    # preallocate
    #out=np.zeros(len(series))*np.nan
    uang=np.radians(np.arange(1,360,res))
    out=np.zeros(len(uang)+1)
    count=0
    for ii in uang:
        delta=np.min(np.column_stack((circ-abs(rad-ii),np.abs(rad-ii))),axis=1)
        logi=delta<=thresh
        out[count]=f(series[logi])
        count+=1
        
    uang=np.degrees(uang)
    uang_out=np.zeros(len(uang)+1); uang_out[:-1]=uang; uang_out[-1]=uang[0]
    out[-1]=out[0]
    return uang_out,out


def circ_correl(series1,series2,wdir,span,res):
    
    """
    Info to follow 
    """
    
    # Radians
    rad=np.radians(wdir)
    thresh=np.radians(span)
    circ=2*np.pi
    
    # preallocate
    uang=np.radians(np.arange(1,360,res))
    out=np.zeros(len(uang)+1)
    nanidx=np.logical_and(~np.isnan(series1),~np.isnan(series2))
    count=0
    for ii in uang:
        delta=np.min(np.column_stack((circ-abs(rad-ii),np.abs(rad-ii))),axis=1)
        #delta=np.array([ circ - np.min([circ-abs(jj-ii),abs(jj-ii)]) for jj in rad])
        logi=delta<=thresh
        logi=np.logical_and(logi,nanidx)
        out[count]=np.corrcoef(series1[logi],series2[logi])[0,1]
        count+=1
        
    uang=np.degrees(uang)
    uang_out=np.zeros(len(uang)+1); uang_out[:-1]=uang; uang_out[-1]=uang[0]
    out[-1]=out[0]
    return uang_out,out

def calc_wdir(u,v):
    
    """
    Calculates the wind direction given u and v components
    """
    wdir = np.arctan2(u,v) * dper + 180.
    return wdir            

# # Manual inspection suggests dubious data quality during this period
st=datetime.datetime(year=2019,month=12,day=27,hour=17)
stp=datetime.datetime(year=2020,month=1,day=2,hour=6)

# Should limit all data to this period
stp_gl=datetime.datetime(year=2020,month=1,day=5,hour=11)

thresh=1 # Filter out hourly-mean winds < this value

# Import the reanalysis
fr="/home/lunet/gytm3/Everest2019/Research/DangerousWinds/Data/SouthCol_interpolated.nc"
figdir="/home/lunet/gytm3/Everest2019/Research/DangerousWinds/Figures/"
r=xr.open_dataset(fr).to_dataframe()
ur=r["ws"]
u_wnd=r["u_wnd"]
v_wnd=r["v_wnd"]
tr=r["temp"]-273.15
pr=r["press"]#
# Wdir from the reanalysis
wdir=calc_wdir(u_wnd,v_wnd)

# Ditto observations -- incl. filtering and alocation
fo="/home/lunet/gytm3/Everest2019/Research/DangerousWinds/Data/south_col.csv"
o=pd.read_csv(fo,parse_dates=True,index_col=0)
for i in o.columns: o[i]=o[i].astype(np.float)
o=o.loc[o.index<=stp_gl]
u=o["WS_AVG_2"]
ug=o["WS_MAX_2"]
t=o["T_HMP"]
p=o["PRESS"]
idx=np.logical_and(u>thresh,ug>thresh) # very minimal filter that removes 
# wind during v. slack conditions
idx=np.logical_and(idx,~np.logical_and(u.index>=st,u.index<=stp))
u.values[~idx]=np.nan
ug.values[~idx]=np.nan
# Read in the horizon angles 
hang=np.loadtxt("/home/lunet/gytm3/Everest2019/Research/DangerousWinds/Data/horz.txt")
hang_plot=np.zeros((len(hang)+1,2)); 
hang_plot[:-1,:]=hang[:,:]
hang_plot[-1,:]=hang[0,:]

# Extract overlapping reanalysis wind
ur_sub=ur.loc[ur.index.isin(u.index)]
wdir_sub=wdir.loc[ur.index.isin(u.index)]
u_sub=u.loc[u.index.isin(ur.index)]
ug_sub=ug.loc[u.index.isin(ur.index)]
# and pressure
pr_sub=p.loc[p.index.isin(ur.index)]

# Compute correlations
idx=np.logical_and(~np.isnan(ur_sub),~np.isnan(ug_sub))
glob_r=np.corrcoef(ur_sub[idx],ug_sub[idx])
rcirc=circ_correl(ug_sub,ur_sub,wdir_sub,15.0,5.)

# Ratio 
rat=ug_sub/ur_sub
rrat=circ_stat(rat,wdir_sub,15,"median",5.)

# Residual
t,p=stats.ttest_ind(ug_sub[idx],ur_sub[idx])
#idx=np.logical_and(idx,np.logical_and(wdir_sub>225,wdir_sub<315))
resid=ug_sub[idx]-ur_sub[idx]; const=np.mean(resid)
resid_new=ug_sub[idx]-(ur_sub[idx]+const)
uncert=np.nanpercentile(np.abs(resid_new),95)
lower_thresh=np.percentile(resid_new[np.argsort(resid_new)],2.5)
upper_thresh=np.percentile(resid_new[np.argsort(resid_new)],97.5)

# t-test
t,p=stats.ttest_ind(ug_sub[idx],ur_sub,equal_var=False)
print("T=%.1f,p=%.3f df=%.0f"%(t,p,2*sum(idx)-2))

# Bias correct?
ur_sub_raw=ur_sub*1.
ur_sub=ur_sub+const
resid=ug_sub[idx]-ur_sub[idx]

# Compute frequency
freq=circ_stat(np.ones(len(wdir_sub)),wdir_sub,45,"sum",45)
freq_pc=freq[1]/np.sum(freq[1])*100

# Draw figures
fig=plt.figure()
fig.set_size_inches(7,6)
ref=np.linspace(0,55,100)
ax1=fig.add_subplot(221)
ax1.scatter(ur_sub,ug_sub,s=1,color='k')
ax1.plot(ref,ref,color='red')
ax1.grid()
ax1.set_xlim([0,55])
ax1.set_ylim([0,55])
ax1.set_ylabel("Obs Gust (m s$^{-1}$)")
ax1.set_xlabel("ERA5 (m s$^{-1}$)")

# Plot horizon angle
ax2=fig.add_subplot(222,projection="polar")
ax2.plot(np.radians(hang_plot[:,0]),hang_plot[:,1],color='k')
ax2.set_theta_direction(-1)
ax2.set_theta_direction(-1)
ax2.set_theta_zero_location("N")
#ax2.set_ylim([0,1.0])

# Plot the circular histogram?
ax2.bar(np.radians(freq[0]),freq_pc,color="grey",linewidth=0.5,edgecolor="red")

# Plot correlation
ax3=fig.add_subplot(223,projection="polar")
ax3.plot(np.radians(rcirc[0]),rcirc[1],color='k')
# PLot ratio
#ax2.plot(np.radians(rrat[0]),rrat[1])
ax3.set_theta_direction(-1)
ax3.set_theta_direction(-1)
ax3.set_theta_zero_location("N")
ax3.set_ylim([0,1.0])
#fig.set_size_inches(4,4)

ax4=fig.add_subplot(224)
ax4.hist(resid,bins=30,facecolor='w',edgecolor='k')
ax4.grid()
ax4.set_xlim([-12,12])
# ax4.axvline(0,linestyle="-",color="red")
ax4.set_xlabel("Residual (m s$^{-1}$)")
ax4.set_ylabel("Frequency (hours)")    
ax4.axvline(lower_thresh,color='red',linestyle="--")
ax4.axvline(upper_thresh,color='red',linestyle="--")
plt.tight_layout()
fig.savefig(figdir+"Reanal_eval.pdf",dpi=300)


# How do the ECDFs look?
ecdf_rc=ur_sub[idx][np.argsort(ur_sub[idx])]
ecdf_r=ur_sub_raw[idx][np.argsort(ur_sub_raw[idx])]
ecdf_g=ug_sub[idx][np.argsort(ug_sub[idx])]
x=np.arange(np.sum(idx))/np.sum(idx).astype(np.float)
fig,ax=plt.subplots(1,2)
fig.set_size_inches(7,3)
ax.flat[0].plot(ecdf_r.values[:],x,color="grey",linewidth=2,label="ERA5")
ax.flat[0].plot(ecdf_g.values[:],x,color="black",linewidth=2,label="Obs Gust")
ax.flat[0].legend()
ax.flat[0].set_ylim([0,1.01])
ax.flat[0].set_xlim([0,55])
ax.flat[0].grid()
ax.flat[0].set_ylabel("ECDF")
ax.flat[0].set_xlabel("Wind Speed (m s$^{-1}$)")

k=ecdf_g-ecdf_r
ax.flat[1].scatter(x,k,color="black",s=1)
ax.flat[1].grid()
ax.flat[1].set_ylabel("k (m s$^{-1}$)")
ax.flat[1].set_xlabel("q")
ax.flat[1].set_xlim([0,1])
ax.flat[1].axhline(const,color='r')
plt.tight_layout()
fig.savefig(figdir+"Reanal_ECDF.pdf",dpi=300)

# Uncertainty range
uncert=np.nanpercentile(resid,95)











