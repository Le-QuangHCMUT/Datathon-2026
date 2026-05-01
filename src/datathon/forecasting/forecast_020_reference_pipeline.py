import os, warnings, pandas as pd, numpy as np, matplotlib.pyplot as plt
import lightgbm as lgb
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

TET = {2012:'2012-01-23',2013:'2013-02-10',2014:'2014-01-31',2015:'2015-02-19',
       2016:'2016-02-08',2017:'2017-01-28',2018:'2018-02-16',2019:'2019-02-05',
       2020:'2020-01-25',2021:'2021-02-12',2022:'2022-02-01',2023:'2023-01-22',
       2024:'2024-02-10',2025:'2025-01-29'}
TET_TS = {k: pd.to_datetime(v) for k, v in TET.items()}

PROMOS = [
    {'name':'spring_sale','m':3,'d':18,'dur':30,'disc':12,'odd':False},
    {'name':'mid_year','m':6,'d':23,'dur':29,'disc':18,'odd':False},
    {'name':'fall_launch','m':8,'d':30,'dur':32,'disc':10,'odd':False},
    {'name':'year_end','m':11,'d':18,'dur':45,'disc':20,'odd':False},
    {'name':'urban_blowout','m':7,'d':30,'dur':33,'disc':0,'odd':True},
    {'name':'rural_special','m':1,'d':30,'dur':30,'disc':15,'odd':True},
]

def build_features(idx):
    feat = pd.DataFrame(index=idx)
    feat['year']   = idx.year
    feat['month']  = idx.month
    feat['day']    = idx.day
    feat['dow']    = idx.dayofweek
    feat['doy']    = idx.dayofyear
    feat['quarter']= idx.quarter
    feat['is_weekend'] = (idx.dayofweek>=5).astype(int)
    feat['is_odd_year'] = (idx.year%2!=0).astype(int)
    feat['days_to_eom']  = idx.days_in_month - idx.day
    feat['days_from_som']= idx.day - 1
    feat['dim'] = idx.days_in_month
    feat['is_last1']  = (feat['days_to_eom']==0).astype(int)
    feat['is_last2']  = (feat['days_to_eom']==1).astype(int)
    feat['is_last3']  = (feat['days_to_eom']<=2).astype(int)
    feat['is_first1'] = (idx.day==1).astype(int)
    feat['is_first2'] = (idx.day==2).astype(int)
    feat['is_first3'] = (idx.day<=3).astype(int)
    ref = pd.Timestamp('2020-01-01')
    feat['t_days']  = (idx - ref).days
    feat['t_years'] = feat['t_days'] / 365.25
    feat['regime_pre2019']  = (idx.year<2019).astype(int)
    feat['regime_2019']     = (idx.year==2019).astype(int)
    feat['regime_post2019'] = (idx.year>=2020).astype(int)
    for k in range(1,6):
        feat[f'sin_y{k}'] = np.sin(2*np.pi*k*idx.dayofyear/365.25)
        feat[f'cos_y{k}'] = np.cos(2*np.pi*k*idx.dayofyear/365.25)
    for k in range(1,3):
        feat[f'sin_w{k}'] = np.sin(2*np.pi*k*idx.dayofweek/7)
        feat[f'cos_w{k}'] = np.cos(2*np.pi*k*idx.dayofweek/7)
        feat[f'sin_m{k}'] = np.sin(2*np.pi*k*idx.day/idx.days_in_month)
        feat[f'cos_m{k}'] = np.cos(2*np.pi*k*idx.day/idx.days_in_month)
    feat['hol_new_year']     = ((idx.month==1)&(idx.day==1)).astype(int)
    feat['hol_womens_day']   = ((idx.month==3)&(idx.day==8)).astype(int)
    feat['hol_reunification']= ((idx.month==4)&(idx.day==30)).astype(int)
    feat['hol_labor_day']    = ((idx.month==5)&(idx.day==1)).astype(int)
    feat['hol_national_day'] = ((idx.month==9)&(idx.day==2)).astype(int)
    feat['hol_vn_womens_day']= ((idx.month==10)&(idx.day==20)).astype(int)
    feat['hol_dd_1111']      = ((idx.month==11)&(idx.day==11)).astype(int)
    feat['hol_dd_1212']      = ((idx.month==12)&(idx.day==12)).astype(int)
    feat['hol_christmas_eve']= ((idx.month==12)&(idx.day==24)).astype(int)
    feat['hol_christmas']    = ((idx.month==12)&(idx.day==25)).astype(int)
    feat['hol_black_friday'] = ((idx.month==11)&(idx.dayofweek==4)&(idx.day>=24)).astype(int)
    # Tet
    yt = idx.year.map(TET_TS)
    yt.index = idx
    yn = (idx.year+1).map(TET_TS); yn.index = idx
    yp = (idx.year-1).map(TET_TS); yp.index = idx
    dt = (idx - yt).days; dn = (idx - yn).days; dp = (idx - yp).days
    diff = pd.Series(dt, index=idx, dtype=float)
    diff[np.abs(dn) < np.abs(diff)] = dn[np.abs(dn) < np.abs(diff)]
    diff[np.abs(dp) < np.abs(diff)] = dp[np.abs(dp) < np.abs(diff)]
    feat['tet_days_diff'] = diff.values
    feat['tet_in_7']      = ((diff>=-7)&(diff<0)).astype(int).values
    feat['tet_in_14']     = ((diff>=-14)&(diff<0)).astype(int).values
    feat['tet_before_7']  = ((diff>=-7)&(diff<0)).astype(int).values
    feat['tet_after_7']   = ((diff>0)&(diff<=7)).astype(int).values
    feat['tet_on']        = (diff==0).astype(int).values
    # Promos
    for p in PROMOS:
        feat[f"promo_{p['name']}"]       = 0
        feat[f"promo_{p['name']}_since"] = 0
        feat[f"promo_{p['name']}_until"] = 0
        feat[f"promo_{p['name']}_disc"]  = 0
        for y in range(2012,2026):
            if p['odd'] and y%2==0: continue
            try: sd = pd.Timestamp(y, p['m'], p['d'])
            except: continue
            ed = sd + pd.Timedelta(days=p['dur']-1)
            mk = (idx>=sd)&(idx<=ed)
            feat.loc[mk, f"promo_{p['name']}"] = 1
            feat.loc[mk, f"promo_{p['name']}_since"] = (idx[mk]-sd).days
            feat.loc[mk, f"promo_{p['name']}_until"] = (ed-idx[mk]).days
            feat.loc[mk, f"promo_{p['name']}_disc"]  = p['disc']
    feat = feat.fillna(0)
    return feat

LGB_PARAMS = dict(objective='regression',metric='mae',learning_rate=0.03,
    num_leaves=63,min_data_in_leaf=30,feature_fraction=0.85,bagging_fraction=0.85,
    bagging_freq=5,lambda_l2=1.0,seed=42,verbosity=-1)

def get_weights(dates, quarter_boost=None, QBOOST=2.0):
    w = np.where((dates.year>=2014)&(dates.year<=2018), 1.0, 0.01)
    if quarter_boost is not None:
        w = np.where(dates.quarter==quarter_boost, w*QBOOST, w)
    return w

def train_lgb(y_tr, X_tr, X_te, dates_tr, quarter_boost=None):
    w = get_weights(dates_tr, quarter_boost)
    split = max(180, len(X_tr)-180)
    Xt, Xv = X_tr.iloc[:split], X_tr.iloc[split:]
    yt, yv = np.log(y_tr.iloc[:split]), np.log(y_tr.iloc[split:])
    wt, wv = w[:split], w[split:]
    dt = lgb.Dataset(Xt,yt,weight=wt)
    dv = lgb.Dataset(Xv,yv,weight=wv,reference=dt)
    m = lgb.train(LGB_PARAMS, dt, 2000, valid_sets=[dt,dv],
                  callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
    best = m.best_iteration
    m2 = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr,np.log(y_tr),weight=w), best)
    return np.exp(m2.predict(X_te)), m2

def train_ridge(y_tr, X_tr, X_te):
    sc = StandardScaler()
    Xts = sc.fit_transform(X_tr)
    Xvs = sc.transform(X_te)
    m = Ridge(alpha=3.0).fit(Xts, np.log(y_tr))
    return np.exp(m.predict(Xvs)), m, sc

def train_prophet_model(y_tr, dates_tr, dates_te, X_tr, X_te):
    mask = dates_tr>='2020-01-01'
    df = pd.DataFrame({'ds':dates_tr[mask],'y':np.log(y_tr[mask].values)})
    pcols = [f"promo_{p['name']}" for p in PROMOS]
    for c in pcols: df[c] = X_tr.loc[mask, c].values
    m = Prophet(yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=False,
                seasonality_mode='multiplicative',changepoint_prior_scale=0.05)
    for c in pcols: m.add_regressor(c)
    m.fit(df)
    df_f = pd.DataFrame({'ds':dates_te})
    for c in pcols: df_f[c] = X_te[c].values
    return np.exp(m.predict(df_f)['yhat'].values)

def main():
    print("FORECAST-020: Reference Pipeline")
    dep = [{'package':'lightgbm','version':lgb.__version__,'status':'OK'},
           {'package':'prophet','version':Prophet.__module__,'status':'OK'}]
    pd.DataFrame(dep).to_csv('artifacts/tables/forecast_020_dependency_check.csv',index=False)

    print("Loading data...")
    sales = pd.read_csv('data/sales.csv')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales = sales.sort_values('Date').set_index('Date')
    sample = pd.read_csv('sample_submission.csv')
    sample['Date'] = pd.to_datetime(sample['Date'])
    fdates = pd.DatetimeIndex(sample['Date'])
    sub012 = pd.read_csv('artifacts/submissions/submission_forecast_012_calibrated.csv')
    sub012['Date'] = pd.to_datetime(sub012['Date']); sub012 = sub012.set_index('Date')
    assert len(sample)==548, "Sample mismatch"
    print(f"Sales: {sales.index.min()} to {sales.index.max()}, {len(sales)} rows")

    print("Building features...")
    feat_all    = build_features(sales.index)
    feat_future = build_features(fdates)
    print(f"Features: {feat_all.shape[1]}")
    assert feat_all.isna().sum().sum()==0, "NaN in train features"
    assert feat_future.isna().sum().sum()==0, "NaN in future features"

    audit = pd.DataFrame({'feature':feat_all.columns,'non_null':feat_all.notna().sum().values})
    audit.to_csv('artifacts/tables/forecast_020_feature_audit.csv',index=False)

    folds = [
        ('FoldA','2021-12-31','2022-01-01','2022-12-31'),
        ('FoldB','2020-12-31','2021-01-01','2021-12-31'),
        ('FoldC','2021-06-30','2021-07-01','2022-06-30'),
    ]

    cv_metrics=[]; base_metrics=[]; ens_metrics=[]

    for fname,tend,vst,vend in folds:
        print(f"Fold {fname}...")
        mtr = sales.index<=tend; mva=(sales.index>=vst)&(sales.index<=vend)
        Xtr,Xva = feat_all[mtr],feat_all[mva]
        yRevTr,yRevVa = sales.loc[mtr,'Revenue'],sales.loc[mva,'Revenue']
        yCogsTr,yCogsVa = sales.loc[mtr,'COGS'],sales.loc[mva,'COGS']
        dtr,dva = sales.index[mtr],sales.index[mva]

        rRidgeRev,_,_ = train_ridge(yRevTr,Xtr,Xva)
        rRidgeCogs,_,_= train_ridge(yCogsTr,Xtr,Xva)
        rLgbRev,_   = train_lgb(yRevTr,Xtr,Xva,dtr)
        rLgbCogs,_  = train_lgb(yCogsTr,Xtr,Xva,dtr)
        rProRev  = train_prophet_model(yRevTr,dtr,dva,Xtr,Xva)
        rProCogs = train_prophet_model(yCogsTr,dtr,dva,Xtr,Xva)

        rQRev=np.zeros(len(Xva)); rQCogs=np.zeros(len(Xva))
        for q in [1,2,3,4]:
            pR,_ = train_lgb(yRevTr,Xtr,Xva,dtr,quarter_boost=q)
            pC,_ = train_lgb(yCogsTr,Xtr,Xva,dtr,quarter_boost=q)
            mk = dva.quarter==q; rQRev[mk]=pR[mk]; rQCogs[mk]=pC[mk]

        base_metrics += [
            {'fold':fname,'target':'Revenue','model':'Ridge','MAE':mean_absolute_error(yRevVa,rRidgeRev)},
            {'fold':fname,'target':'Revenue','model':'LGB',  'MAE':mean_absolute_error(yRevVa,rLgbRev)},
            {'fold':fname,'target':'Revenue','model':'Prophet','MAE':mean_absolute_error(yRevVa,rProRev)},
            {'fold':fname,'target':'Revenue','model':'QSpec','MAE':mean_absolute_error(yRevVa,rQRev)},
            {'fold':fname,'target':'COGS',   'model':'Ridge','MAE':mean_absolute_error(yCogsVa,rRidgeCogs)},
            {'fold':fname,'target':'COGS',   'model':'LGB',  'MAE':mean_absolute_error(yCogsVa,rLgbCogs)},
            {'fold':fname,'target':'COGS',   'model':'Prophet','MAE':mean_absolute_error(yCogsVa,rProCogs)},
            {'fold':fname,'target':'COGS',   'model':'QSpec','MAE':mean_absolute_error(yCogsVa,rQCogs)},
        ]

        if fname=='FoldA':
            rdf = pd.DataFrame({'Ridge':rRidgeRev-yRevVa.values,'LGB':rLgbRev-yRevVa.values,
                                'Prophet':rProRev-yRevVa.values,'QSpec':rQRev-yRevVa.values})
            rdf.corr().to_csv('artifacts/tables/forecast_020_residual_correlation.csv')
            plt.figure(figsize=(8,6)); plt.imshow(rdf.corr(),aspect='auto')
            plt.title('Residual Correlation'); plt.savefig('artifacts/figures/forecast_020_residual_correlation.png'); plt.close()

        for alpha in [0.45,0.50,0.55,0.60,0.65,0.70]:
            blendR=alpha*rQRev+(1-alpha)*rLgbRev
            blendC=alpha*rQCogs+(1-alpha)*rLgbCogs
            rawR=0.10*rRidgeRev+0.10*rProRev+0.80*blendR
            rawC=0.10*rRidgeCogs+0.10*rProCogs+0.80*blendC
            for cr in [1.18,1.22,1.26,1.30,1.34]:
                for cc in [1.28,1.30,1.32,1.34,1.36]:
                    ens_metrics.append({'fold':fname,'alpha':alpha,'cr':cr,'cc':cc,
                        'RevMAE':mean_absolute_error(yRevVa,rawR*cr),
                        'CogsMAE':mean_absolute_error(yCogsVa,rawC*cc)})

    pd.DataFrame(base_metrics).to_csv('artifacts/tables/forecast_020_base_model_metrics.csv',index=False)
    emdf=pd.DataFrame(ens_metrics)
    emdf.to_csv('artifacts/tables/forecast_020_cv_metrics.csv',index=False)
    emdf.groupby(['alpha','cr','cc'])[['RevMAE','CogsMAE']].mean().to_csv('artifacts/tables/forecast_020_ensemble_grid_metrics.csv')

    plt.figure(figsize=(12,5))
    bm=pd.DataFrame(base_metrics)
    for tgt,ax in zip(['Revenue','COGS'],[plt.subplot(1,2,1),plt.subplot(1,2,2)]):
        bm[bm.target==tgt].groupby('model')['MAE'].mean().plot(kind='bar',ax=ax,title=tgt)
    plt.tight_layout(); plt.savefig('artifacts/figures/forecast_020_model_comparison.png'); plt.close()

    print("Training final models on full history...")
    fRevRidge,_,_ = train_ridge(sales['Revenue'],feat_all,feat_future)
    fCogRidge,_,_ = train_ridge(sales['COGS'],feat_all,feat_future)
    fRevLgb,lgbRM = train_lgb(sales['Revenue'],feat_all,feat_future,sales.index)
    fCogLgb,_     = train_lgb(sales['COGS'],feat_all,feat_future,sales.index)
    fRevPro  = train_prophet_model(sales['Revenue'],sales.index,fdates,feat_all,feat_future)
    fCogPro  = train_prophet_model(sales['COGS'],sales.index,fdates,feat_all,feat_future)

    fRevQ=np.zeros(len(fdates)); fCogQ=np.zeros(len(fdates))
    for q in [1,2,3,4]:
        pR,_=train_lgb(sales['Revenue'],feat_all,feat_future,sales.index,quarter_boost=q)
        pC,_=train_lgb(sales['COGS'],feat_all,feat_future,sales.index,quarter_boost=q)
        mk=fdates.quarter==q; fRevQ[mk]=pR[mk]; fCogQ[mk]=pC[mk]

    lgb.plot_importance(lgbRM,max_num_features=20); plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_020_feature_importance_lgb.png'); plt.close()

    ALPHA=0.60
    blendR=ALPHA*fRevQ+(1-ALPHA)*fRevLgb
    blendC=ALPHA*fCogQ+(1-ALPHA)*fCogLgb
    rawR=0.10*fRevRidge+0.10*fRevPro+0.80*blendR
    rawC=0.10*fCogRidge+0.10*fCogPro+0.80*blendC

    fdf=pd.DataFrame({'Date':fdates.values,'rawRev':rawR,'rawCog':rawC,
                      'Ridge_Rev':fRevRidge,'LGB_Rev':fRevLgb,'Prophet_Rev':fRevPro,'QSpec_Rev':fRevQ,
                      'blend_Rev':blendR,'blend_Cog':blendC})
    fdf.to_csv('artifacts/forecasts/forecast_020_future_predictions.csv',index=False)

    def save_sub(name,rev,cogs):
        d=pd.DataFrame({'Date':fdates.values,'Revenue':np.maximum(rev,1),'COGS':np.maximum(cogs,0)})
        assert len(d)==548 and d.isna().sum().sum()==0
        d.to_csv(f'artifacts/submissions/submission_forecast_020_{name}.csv',index=False)
        return d

    sA=save_sub('ref_alpha060_cr126_cc132',rawR*1.26,rawC*1.32)
    sB=save_sub('ref_alpha060_cr122_cc130',rawR*1.22,rawC*1.30)
    sC=save_sub('ref_alpha060_cr130_cc134',rawR*1.30,rawC*1.34)
    sD=save_sub('ref_blend012_25',0.75*sub012['Revenue'].values+0.25*sA['Revenue'].values,
                                  0.75*sub012['COGS'].values+0.25*sA['COGS'].values)
    sE=save_sub('ref_blend012_50',0.50*sub012['Revenue'].values+0.50*sA['Revenue'].values,
                                  0.50*sub012['COGS'].values+0.50*sA['COGS'].values)

    plt.figure(figsize=(15,6))
    plt.plot(fdates,sA['Revenue'],label='Ref CR1.26'); plt.plot(fdates,sB['Revenue'],label='Ref CR1.22')
    plt.plot(fdates,sub012['Revenue'],label='012 base',alpha=0.5); plt.legend()
    plt.title('Forecast 020 Future Profiles'); plt.savefig('artifacts/figures/forecast_020_future_profiles.png'); plt.close()

    best_ens = emdf.groupby(['alpha','cr','cc'])[['RevMAE','CogsMAE']].mean()
    best_ens['combined'] = best_ens['RevMAE']+best_ens['CogsMAE']

    def pct_vs012(rev): return (rev.sum()-sub012['Revenue'].sum())/sub012['Revenue'].sum()*100
    def mad_vs012(rev): return (np.abs(rev-sub012['Revenue'].values)/sub012['Revenue'].values).mean()*100

    manifest=[]
    for nm,sub,cr,cc,pri,submit in [
        ('ref_alpha060_cr126_cc132',sA,1.26,1.32,1,'submit'),
        ('ref_alpha060_cr122_cc130',sB,1.22,1.30,2,'hold'),
        ('ref_alpha060_cr130_cc134',sC,1.30,1.34,3,'hold'),
        ('ref_blend012_25',sD,1.26,1.32,4,'hold'),
        ('ref_blend012_50',sE,1.26,1.32,5,'hold'),
    ]:
        ref_row = best_ens.loc[(0.60,cr,cc)] if (0.60,cr,cc) in best_ens.index else best_ens.iloc[0]
        manifest.append({'candidate_file':f'submission_forecast_020_{nm}.csv','candidate_name':nm,
            'method':'Reference Pipeline','dependencies_available':True,'is_true_reference_pipeline':True,
            'alpha':0.60,'cr':cr,'cc':cc,'CV_Revenue_MAE':ref_row['RevMAE'],'CV_COGS_MAE':ref_row['CogsMAE'],
            'CV_combined_MAE':ref_row['combined'],'total_revenue_forecast':sub['Revenue'].sum(),
            'total_cogs_forecast':sub['COGS'].sum(),'pct_diff_revenue_vs_012':pct_vs012(sub['Revenue']),
            'mean_abs_pct_diff_daily_vs_012':mad_vs012(sub['Revenue']),'expected_public_risk':'medium',
            'recommended_submission_priority':pri,'submit_or_hold':submit})

    pd.DataFrame(manifest).to_csv('artifacts/tables/forecast_020_candidate_manifest.csv',index=False)
    print("Done.")

if __name__=='__main__':
    main()
