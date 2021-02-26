#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""
import xarray as xr, pandas as pd, numpy as np, matplotlib.pyplot as plt,\
datetime,calendar
from sklearn import linear_model
from matplotlib.pyplot import cm
from numba import jit
from scipy import stats
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
dper =180./np.pi # Degrees per radian

# # Manual inspection suggests dubious data quality during this period
st=datetime.datetime(year=2019,month=12,day=27,hour=17)
stp=datetime.datetime(year=2020,month=1,day=2,hour=6)

# Should limit all data to this period
stp_gl=datetime.datetime(year=2020,month=1,day=5,hour=11)

thresh=1 # Filter out hourly-mean winds < this value

# Functions
def RHO(p,tv):
    
    """
    Computes the air density
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - p (Pa)        : air pressure
        - tv (K or C)   : virtual temperature (can be approximated as T if air 
                          is very dry -- low specific humidity)
        
    Out:
    
        - rho (kg/m^3) : air density
        
    """ 
    _rd=287.053 # Gas constant for dry air
    _tv=tv*1.
    if np.nanmax(_tv)<100: _tv +=273.15# NB: C-->K
    if np.nanmax(p)<2000: p*=100 # hPa to Pa
    rho=np.divide(p,np.multiply(_rd,_tv))

    return rho

def CRIT_U(p,tv):
    
    """
    Computes the wind speed required to yield a force of 72 N
    on the "average" man. 
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - p (Pa)       : air pressure
        - tv (K or C)  : virtual temperature (can be approximated as T if air is
                         very dry -- low specific humidity)
        
    Out:
    
        - uc (m/s)     : wind speed to produce force of 72N
        
    """    
    rho=RHO(p,tv)
    # Note: 144 = 2x 72 N; 0.3 = 0.6 drag coef * surface area 0.5 m**2
    uc=np.power(np.divide(144,np.multiply(rho,0.3)),0.5)
    
    return uc


def seas_cycle(ind,n,indata,ncycles=1):
    
    """
    Fits the seasonal cycle using harmonic
    regression.
    
    Inputs/Outputs (units: explanation): 
    
    In:
        - ind          : index variable. 
        - n            : number of obs per cycle
        - indata       : variable to compute seasonal
                         cycle for!
        - ncycles      : frequency 
        
    Out:
    
        - uc (m/s)     : seasonal cycle for indata
        
    """  
    y=indata.values[:]
    reg = linear_model.LinearRegression() 
    ind = 2*np.pi*(ind/np.float(n)) 
    idx = ~np.isnan(y)
    cosines=np.zeros((len(ind),ncycles))*np.nan
    sines=np.zeros(cosines.shape)*np.nan
    for i in np.arange(1,ncycles+1):
        cosines[idx,np.int(i-1)]=np.cos(ind[idx]*i)
        sines[idx,np.int(i-1)]=np.sin(ind[idx]*i)
    X = np.column_stack((cosines,sines));
    reg.fit(X[idx,:], y[idx]) 
    sim = reg.predict(X)
    return sim

def himDate(datecol,hourcol,verb=False):
    
    """
    Simply iterates over datestrings (day/month/year)
    and returns a vector of datetimes
    
    In:
        - datecol     : of form "day/month/year"
        - hourcol     : decimal form
        - verb        : if verb, print outcome from except
        
    Out:
    
        - uc (m/s)    : datetime.datetime
        
    """
    date=[]
    for i in range(len(datecol)):
        try:
            day=np.int(datecol.values[i][:2])
            month=np.int(datecol.values[i][3:5])
            year=np.int(datecol.values[i][6:])
            hour=np.int(np.round(hourcol.values[i])); 
            if hour == 0: hour= 6# Do this to force to middle of calendar day (needed if no hour recorded)
            d=datetime.datetime(year=year,month=month,day=day,hour=hour)
            
        except: 
            d=np.nan
            if verb:
                print(datecol.values[i])

        date.append(d)
        
    return date
        
    
def match_winds(dates,wind_df,stat,tol_hours,shift_time=0):
            
        """
        Iterate over all dates in 'dates' and
        find all wind observations within +/- 
        (tol_hours+shift_time)

        In:
            - dates     : datetime object for which to find winds
            - wind_df   : series with winds (and time index)
            - stat      : statistic to compute (string -- from numpy)
            - tol_hours : stat wind taken for +/- this many hours 
            - shift_time: n hours to adjust to the same time zone
                         (will be added on to the dates)

        Out:

            - df_out    : dataframe with max wind within tol_hours 
                          of each date in dates 

        """
        # Substitutions
        f=getattr(np,stat)
        wtime=wind_df.index
        delta=datetime.timedelta(hours=tol_hours)
        out=np.zeros(len(dates))*np.nan
        count=0
        for d in dates:
            if isinstance(d,float): count +=1; continue
            di=d+datetime.timedelta(hours=shift_time)
            idx=np.logical_and(wtime>=(di-delta),wtime<=(di+delta))
            if np.sum(idx)>0:
                out[count]=f(wind_df.loc[idx])
            count+=1
            
        out=pd.Series(out,index=dates)
        return out
    
    
    
def clusters(dates,threshold,max_ht):
 
    """
    Iterate over all dates in 'dates' and
    find all wind observations within +/- 
    (tol_hours+shift_time)

    In:
        - dates     : datetime object 
        - threshold : ndays: all dates within this ndays of another
                      will be treated as non-unique
        - max_ht    : maximum elevation reached by the climbers


    Out:

        - id        : ids of the different clusters
        - centre    : centre points/dates for the clusters
    """
    dates=np.array(dates)
    const=24*60**2
    id=np.zeros(len(dates))*np.nan
    idi=0
    count=0
    for i in range(len(dates)):
        if count == 0: id[count]=idi; count +=1; continue       
        # create temporary indexing var
        inds=np.arange(len(dates))
        # Drop var for this iteration
        inds=inds[inds != i]
        # Difference with all others
        dm=np.abs((dates[i]-dates[inds])/np.timedelta64(1,'s')/const)
        # Take min. 
        
        # If yes, we have another match somewhere     
        if np.min(dm) <=threshold:          
            nbs = id[inds]
            nbs = nbs[dm<=threshold]
            nbs_valid=nbs[~np.isnan(nbs)]
            # Have a neighbour already allocated 
            # an id
            if len(nbs_valid)>=1:
                id[count]=nbs_valid[0]
            # Neighbours, but no id yet
            else: 
                idi+=1
                id[count] = idi       
                
        else:            
            idi+=1
            id[count] = idi
        count +=1
    udates=[]  
    meta={}
    maxs={}
    for i in np.unique(id):
        a=np.min(dates[id==i])
        b=np.max(dates[id==i])
        mid=a+(b-a)/2
        udates.append(mid)
        meta[mid]=dates[id==i]
        maxs[mid]=np.max(max_ht[id==i])
        
        
    return udates, meta, maxs

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
    out=np.zeros(len(uang))
    count=0
    for ii in uang:
        delta=np.min(np.column_stack((circ-abs(rad-ii),np.abs(rad-ii))),axis=1)
        logi=delta<=thresh
        out[count]=f(series[logi])
        count+=1
        
    uang=np.degrees(uang)
    return uang,out


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
    out=np.zeros(len(uang))
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
    return uang,out

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

# Extract overlapping reanalysis wind
ur_sub=ur.loc[ur.index.isin(u.index)]
wdir_sub=wdir.loc[ur.index.isin(u.index)]
# and obs
u_sub=u.loc[u.index.isin(ur.index)]
ug_sub=ug.loc[u.index.isin(ur.index)]
# and pressure
pr_sub=pr.loc[pr.index.isin(u.index)]
p_sub=p.loc[p.index.isin(pr.index)]
# and temp
tr_sub=tr.loc[tr.index.isin(t.index)]
t_sub=t.loc[t.index.isin(tr.index)]

# Redefine index and non-nan wind
idx=np.logical_and(~np.isnan(ur_sub),~np.isnan(ug_sub))
# Define correction factor 
cf=np.mean(ug_sub[idx]-ur_sub[idx])
# Correct
recon=ur+cf
# Write out
recon.to_csv("Reconstructed.csv")
# Get errors 
error=ur_sub[idx]+cf-ug_sub[idx]
error=error[np.argsort(error)]
lower_thresh=np.percentile(error,2.5)
upper_thresh=np.percentile(error,97.5)
lower=recon+lower_thresh
upper=recon+upper_thresh

# Create figure -- wind speeds on polar plot
fig=plt.figure()
fig.set_size_inches(5,5)
ax=fig.add_subplot(111,projection="polar")
ax.set_theta_direction(-1)
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")
ymax=recon.groupby(recon.index.dayofyear).max()
ymin=recon.groupby(recon.index.dayofyear).min()
ymean=recon.groupby(recon.index.dayofyear).mean()
x_act=ug.index.dayofyear/365.*np.pi*2.
ax.scatter(x_act,ug,color='k',linewidth=0.4,s=1)
x=np.arange(1,367)/366.*2*np.pi
ax.fill_between(x,ymin.values[:],ymax.values[:],alpha=0.4)
ax.plot(x,ymin.values[:],color='r',linewidth=0.5)
ax.plot(x,ymax.values[:],color='r',linewidth=0.5)
ax.plot(x,np.ones(len(x))*30.,color='red')
seas_mean=seas_cycle(np.arange(len(ymean)),366.,ymean,ncycles=4)
ax.plot(x,seas_mean,color='b',linewidth=1,linestyle='-')
xmons=np.arange(181,275)/366.*np.pi*2.
ax.fill_between(xmons,np.zeros(len(xmons)),np.ones(len(xmons))*70,color='k',\
                alpha=0.2)

xticks=(np.array([1,32,60,91,121,152,182,213,244,274,305,335])-1)/366.*np.pi*2
xlabs=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabs)
ax.set_ylim([0,70])
fig.savefig(figdir+"Recon.tif",dpi=300)

# Create table
out=np.zeros(12)
for m in range(12):
    out[m]=np.sum(recon.loc[recon.index.month==(m+1)]>30)/\
    np.sum(recon.index.month==(m+1)).astype(np.float)*100.
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Now extract reconstructed wind speeds for the climbs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    
# fin="/home/lunet/gytm3/Everest2019/Research/Weather/Data/himdata_data_updated.csv"
fin="/home/lunet/gytm3/Everest2019/Research/DangerousWinds/Data/log.txt"
climbs=pd.read_csv(fin,sep="#")
climbs["deathhgtm"]=climbs["deathhgtm"].astype(np.float)
# Extract summits and get dates into proper form
summits=climbs.loc[np.logical_and(climbs["success"]==1,climbs["death"]==0)]
summit_dates=himDate(summits["msmtdate"],climbs["dsmttime"],verb=False)
deaths=climbs.loc[climbs["deathhgtm"]>=8000] # High deaths only
death_dates=himDate(deaths["deathdate"],deaths["ddthtime"],verb=False) # Dates of deaths
missing=climbs.loc[np.logical_and(climbs["deathhgtm"]>=8000,climbs["deathtype"]==9)] # Deathtype = missing
missing_dates=himDate(missing["deathdate"],missing["ddthtime"],verb=True) # Dates of deaths
# Fall in bad weather 
fall_weather=climbs.loc[np.logical_and(climbs["deathhgtm"]>=8000,\
                np.logical_and(climbs["deathtype"]==4,climbs["weather"]==1))]    
fall_weather_dates=himDate(fall_weather["deathdate"],fall_weather["ddthtime"],\
                           verb=False) # Dates of fall+ weather deaths
# Falls
fall=climbs.loc[np.logical_and(climbs["deathhgtm"]>=8000,climbs["deathtype"]==4)]
fall_dates=himDate(fall["deathdate"],fall["ddthtime"],verb=False)
# High points
highs=climbs.loc[np.logical_and(climbs["death"]==0,climbs["mperhighpt"]>=8000)]
high_dates=himDate(highs["msmtdate"],highs["dsmttime"],verb=False)
# High fails, but survived
turns=climbs.loc[np.logical_and(np.logical_and(climbs["mperhighpt"]>=8000,\
                            climbs["mperhighpt"]<8848),climbs["death"]==0)]
turn_dates=himDate(turns["msmtdate"],turns["dsmttime"],verb=False)
# Simply all above
all_above=climbs.loc[climbs["mperhighpt"]>=8000]
all_above_dates=himDate(all_above["msmtdate"],all_above["dsmttime"],verb=False)
# weather death
weather_deaths=deaths.loc[deaths["weather"]==1] # Weather contributed to the death
weather_dates=himDate(weather_deaths["deathdate"],weather_deaths["ddthtime"],verb=False) # Dates of deaths

# Atribute wind (max gust +/- 12 hours from time) during summit
summit_winds=match_winds(summit_dates,recon,"mean",12,shift_time=-6)

# High winds -- ditto, but don't need to have summited. 
high_winds=match_winds(high_dates,recon,"mean",12,shift_time=-6)

# Winds for those that turned around
turn_winds=match_winds(turn_dates,recon,"mean",12,shift_time=-6)

# Ditto, but deaths
death_winds=match_winds(death_dates,recon,"mean",12,shift_time=-6)

# Ditto, death by missing 
missing_winds=match_winds(missing_dates,recon,"mean",12,shift_time=-6)

# Ditto, death by fall when weather was bad
fall_weather_winds=match_winds(fall_weather_dates,recon,"mean",12,shift_time=-6)

# Ditto, but just a fall (not necessarily bad weather)
fall_winds=match_winds(fall_dates,recon,"mean",12,shift_time=-6)

# Ditto, but all those above -- alive or dead
all_above_winds=match_winds(all_above_dates,recon,"mean",12,shift_time=-6)


# Clean all -- remove NaN and drop duplicates
high_winds=high_winds.dropna().sort_index().drop_duplicates()
fall_winds=fall_winds.dropna().sort_index().drop_duplicates()
death_winds=death_winds.dropna().sort_index().drop_duplicates()
all_above_winds=all_above_winds.dropna().sort_index().drop_duplicates()
turn_winds=turn_winds.dropna().sort_index().drop_duplicates()
summit_winds=summit_winds.dropna().sort_index().drop_duplicates()


# Plots and climatology 
# Probability of danger
idx=recon>30
dang=np.ones(len(idx))
ref=dang*1.
dang[~idx]=0
dang=pd.Series(dang,index=recon.index)
ref=pd.Series(ref,index=recon.index)
dang_doy=dang.groupby(dang.index.dayofyear).sum()/\
(ref.groupby(ref.index.dayofyear).sum().astype(np.float))*100.
seas_prob=seas_cycle(np.arange(len(dang_doy)),366.,dang_doy,ncycles=4)

fig=plt.figure()
fig.set_size_inches(8,4)
ax=fig.add_subplot(121,projection="polar")
ax.set_theta_direction(-1)
ax.set_theta_direction(-1)
ax.set_theta_zero_location("N")
x=summit_winds.index.dayofyear/366.*np.pi*2
a1=ax.scatter(x,summit_winds.values[:],color='green',\
   alpha=0.3, s=summit_winds.values[:]/np.nanmax(summit_winds)*50,label="Summited")
x=turn_winds.index.dayofyear/366.*np.pi*2
a2=ax.scatter(x,turn_winds.values[:],color='blue',alpha=0.3,\
              s=turn_winds.values[:]/np.nanmax(turn_winds)*50,label="Turned")
x=fall_winds.index.dayofyear/366.*np.pi*2
a3=ax.scatter(x,fall_winds.values[:],color='red',alpha=1,\
               s=fall_winds.values[:]/np.nanmax(fall_winds)*50,label="Fall Death")
x=missing_winds.index.dayofyear/366.*np.pi*2
a4=ax.scatter(missing_winds.index.dayofyear,missing_winds.values[:],color='orange',\
              alpha=1,s=missing_winds.values[:]/np.nanmax(missing_winds)*50,\
              label="Missing")
ax.plot(np.arange(1,367)/366.*np.pi*2,np.ones(366)*30,color='r')
ax.set_ylim([0,70])
ax.set_xticks(xticks)
ax.set_xticklabels(xlabs)
ax.legend(loc='best', bbox_to_anchor=(0.5, 0.1, 0.2, 0.2))

# Plot those refs on top
x_ref=np.arange(1,367)/366.*(np.pi*2)
ax.plot(x_ref,seas_prob,linestyle='--',color='k')
ax.plot(x_ref,seas_mean,color='k')

# PLot histograms -- difference from "normal"
resid_all=all_above_winds-np.mean(recon)
resid_matched=np.zeros(len(resid_all))
for i in range(len(resid_all)):
    resid_matched[i]=all_above_winds[i]-\
    seas_mean[all_above_winds.index.dayofyear[i]-1]
tout,pout=stats.ttest_1samp(resid_matched,popmean=0)
print("T=%.1f,p=%.3f df=%.0f"%(tout,pout,len(resid_matched)-1))

ax=fig.add_subplot(122)
ax.hist(resid_all,bins=30,edgecolor="k",label='All')
ax.hist(resid_matched,bins=30,alpha=0.4,edgecolor='k',label="Day of year")
ax.axvline(np.mean(resid_all),linewidth=3,color='k',linestyle='--')
ax.axvline(np.mean(resid_matched),linewidth=3,color='k',linestyle='--')
ax.grid()
ax.legend()
ax.set_xlabel("Residual ( m s$^{-1}$)")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(figdir+"Mountaineers_Wind.pdf",dpi=300)
#resid_paired=recon

# Examine the (4) individual climbs in detail 
# For the 1982-12-27/8 climb we have two entries: only take one
excl_date=datetime.datetime(year=1982,month=12,day=28,hour=6)
select=all_above_winds.loc[all_above_winds >= 35]

# Iterate over the dates in 'select' and compute the 24-hour mean centred
# on each time
sub=match_winds(select.index,recon,"mean",12,shift_time=-6)
for i in sub.index:
    mu=np.mean(sub.loc[sub.index==i])
    print("At time:",i," 24-hour mean is %.2f [%.2f-%.2f] ms/"\
         %(mu,mu+lower_thresh,mu+upper_thresh))
    

select=select.append(all_above_winds.loc[all_above_winds.index==\
                                  datetime.datetime(year=1993,\
                                  month=12,day=22,hour=6)])
select=select.append(all_above_winds.loc[all_above_winds.index==\
                               datetime.datetime(year=1993,\
                               month=12,day=18,hour=15)])    
select=select.loc[select.index!=excl_date]
select=select.sort_index()
td=datetime.timedelta(days=1)
fig,ax=plt.subplots(2,2)
fig.set_size_inches(9,5)
fmt=mdates.DateFormatter("%d/%m/%y T%H")

# These dates are mentioned in the text -- they refer to narrow windows
# of time mentioned in the literature. These are needed for plotting
key_dates=[[datetime.datetime(year=1982,month=12,day=27,hour=16,minute=0),\
            datetime.datetime(year=1982,month=12,day=27,hour=20,minute=0)],\
            [],\
            [],\
            [datetime.datetime(year=1993,month=12,day=18,hour=13,minute=0),\
             datetime.datetime(year=1993,month=12,day=18,hour=17,minute=0)],            
            [datetime.datetime(year=1993,month=12,day=20,hour=9,minute=0),\
             datetime.datetime(year=1993,month=12,day=20,hour=13,minute=00)],
            [datetime.datetime(year=1993,month=12,day=22,hour=10,minute=0),\
             datetime.datetime(year=1993,month=12,day=22,hour=12,minute=0)]]

# Having defined them above, iterate over them now to summarise and plot
count=0
reconNPT=recon.copy()
reconNPT.index=recon.index+datetime.timedelta(hours=6) # Convert to NPT
for i in range(len(select)):
    st=select.index[i]-td
    stp=select.index[i]+td
    y=reconNPT.loc[st:stp]
    y_lower=y+lower_thresh
    y_upper=y+upper_thresh
    # print("For 24-hour window starting ",st,
    #           " mean was %.2f [%.2f-%.2f] m/s\n"%(mu,
    #             mu+lower_thresh,mu+upper_thresh))    

    if i == 0 or i >2:
        axi=ax.flat[count]
        axi.plot(y.index[:],y.values[:],color="k")
        axi.fill_between(y.index[:],y.values[:]+lower_thresh,y.values[:]\
               +upper_thresh,color='grey',alpha=0.25)
        axi.set_xlim([y.index.min(),y.index.max()])
        ticks=axi.get_xticks()
        ticks=ticks[1::3]
        axi.set_xticks(ticks)
        axi.xaxis.set_major_formatter(fmt)
        axi.grid()
        axi.set_ylim([0,55])
        axi.axhline(30,color='red')
        axi.axvspan(key_dates[i][0],key_dates[i][1],color="k",alpha=0.5)
        if count == 0 or count == 2: axi.set_ylabel("Wind Gust (m s$^{-1}$)")
        key_y=reconNPT.loc[np.logical_and(reconNPT.index>=key_dates[i][0],
                          reconNPT.index<=key_dates[i][1])]
        mu=np.mean(key_y)
        print("For key date starting ",key_dates[i][0],
              " mean was %.2f [%.2f-%.2f] m/s\n"%(mu,
                mu+lower_thresh,mu+upper_thresh))

        count+=1

fig.savefig(figdir+"Query_gusts.pdf",dpi=300)

# Additional query dates for the 1982 climb
print("Deadly event follows...\n\n")
key_dates2=[[datetime.datetime(year=1982,month=12,day=27,hour=15),
             datetime.datetime(year=1982,month=12,day=27,hour=16)],
            [datetime.datetime(year=1982,month=12,day=27,hour=19),
            datetime.datetime(year=1982,month=12,day=27,hour=20)],
            [datetime.datetime(year=1982,month=12,day=27,hour=20),
            datetime.datetime(year=1982,month=12,day=28,hour=7)]]

for i in range(len(key_dates2)):
    key_y=reconNPT.loc[np.logical_and(reconNPT.index>=key_dates2[i][0],
                              reconNPT.index<=key_dates2[i][1])]
    mu=np.mean(key_y)
    mx=np.max(key_y)
    print("For key date starting ",key_dates2[i][0],
              " mean was %.2f [%.2f-%.2f] m/s, max was %.2f [%.2f-%.2f]\n"%(mu,
                mu+lower_thresh,mu+upper_thresh,mx,mx+lower_thresh,mx+upper_thresh))
    
# Additional query dates for 19th May (2009 - Crampton)
print("2009 event follows...")
key_dates3=[[datetime.datetime(year=2009,month=5,day=19,hour=0),
              datetime.datetime(year=2009,month=5,day=20,hour=0)]]
key_y=reconNPT.loc[np.logical_and(reconNPT.index>=key_dates3[0][0],
                              reconNPT.index<=key_dates3[0][1])]
mu=np.mean(key_y)
mx=np.max(key_y)
print("For key date starting ",key_dates3[0][0],
              " mean was %.2f [%.2f-%.2f] m/s, max was %.2f [%.2f-%.2f]\n"%(mu,
                mu+lower_thresh,mu+upper_thresh,mx,mx+lower_thresh,mx+upper_thresh))

# Additional query dates for 19th May (2019 - Mr Lawless)
print("2019 event follows...")
key_dates3=[[datetime.datetime(year=2019,month=5,day=16,hour=0),
              datetime.datetime(year=2019,month=5,day=17,hour=0)]]
key_y=reconNPT.loc[np.logical_and(reconNPT.index>=key_dates3[0][0],
                              reconNPT.index<=key_dates3[0][1])]
mu=np.mean(key_y)
mx=np.max(key_y)
print("For key date starting ",key_dates3[0][0],
              " mean was %.2f [%.2f-%.2f] m/s, max was %.2f [%.2f-%.2f]\n"%(mu,
                mu+lower_thresh,mu+upper_thresh,mx,mx+lower_thresh,mx+upper_thresh))

# Ditto 1996 event
print("1996 event follows...")
key_dates4=[[datetime.datetime(year=1996,month=5,day=10,hour=0),
              datetime.datetime(year=1996,month=5,day=10,hour=23)]]
key_y=reconNPT.loc[np.logical_and(reconNPT.index>=key_dates4[0][0],
                              reconNPT.index<=key_dates4[0][1])]
mu=np.mean(key_y)
mx=np.max(key_y)
print("For key date starting ",key_dates4[0][0],
              " mean was %.2f [%.2f-%.2f] m/s, max was %.2f [%.2f-%.2f]\n"%(mu,
                mu+lower_thresh,mu+upper_thresh,mx,mx+lower_thresh,mx+upper_thresh))