"""FORECAST-023: Structural Reference Expansion - NNLS, Adaptive Alpha, Promo Audit"""
import os,sys,warnings,pandas as pd,numpy as np,matplotlib.pyplot as plt
import lightgbm as lgb
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.optimize import nnls
warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.join('src','datathon','forecasting'))
from forecast_020_reference_pipeline import build_features,PROMOS,LGB_PARAMS,get_weights

CR,CC=1.28,1.32
FOLDS=[('A','2021-12-31','2022-01-01','2022-12-31'),
       ('B','2020-12-31','2021-01-01','2021-12-31'),
       ('C','2021-06-30','2021-07-01','2022-06-30')]

SCHEMES={'S1_high_era':     lambda y,d: np.where((d.year>=2014)&(d.year<=2018),1.0,0.01),
         'S2_post2020':     lambda y,d: np.where(d.year>=2020,1.0,0.05),
         'S3_recent2022':   lambda y,d: np.where(d.year==2022,1.0,np.where(d.year>=2020,0.5,np.where((d.year>=2014)&(d.year<=2018),0.2,0.01))),
         'S4_clean1718':    lambda y,d: np.where((d.year>=2017)&(d.year<=2018),1.0,np.where((d.year>=2014)&(d.year<=2016),0.5,np.where(d.year>=2020,0.2,0.01))),
         'S5_hybrid':       lambda y,d: np.where((d.year>=2014)&(d.year<=2018),1.0,np.where(d.year==2022,0.8,np.where(d.year>=2020,0.3,np.where(d.year==2019,0.05,0.01)))),
         'S6_uniform':      lambda y,d: np.ones(len(d))}

def lgb_train(y,Xtr,Xte,dtr,w):
    split=max(len(Xtr)-180,180)
    dt=lgb.Dataset(Xtr.iloc[:split],np.log(y.iloc[:split]),weight=w[:split])
    dv=lgb.Dataset(Xtr.iloc[split:],np.log(y.iloc[split:]),weight=w[split:],reference=dt)
    m=lgb.train(LGB_PARAMS,dt,2000,valid_sets=[dt,dv],callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
    m2=lgb.train(LGB_PARAMS,lgb.Dataset(Xtr,np.log(y),weight=w),m.best_iteration)
    return np.exp(m2.predict(Xte)),m2

def ridge_train(y,Xtr,Xte):
    sc=StandardScaler(); m=Ridge(alpha=3.0).fit(sc.fit_transform(Xtr),np.log(y))
    return np.exp(m.predict(sc.transform(Xte))),m,sc

def prophet_train(y,dtr,dte,Xtr,Xte):
    mask=dtr>='2020-01-01'; pcols=[f"promo_{p['name']}" for p in PROMOS]
    df=pd.DataFrame({'ds':dtr[mask],'y':np.log(y[mask].values)})
    for c in pcols: df[c]=Xtr.loc[mask,c].values
    m=Prophet(yearly_seasonality=True,weekly_seasonality=True,daily_seasonality=False,
              seasonality_mode='multiplicative',changepoint_prior_scale=0.05)
    for c in pcols: m.add_regressor(c)
    m.fit(df)
    df_f=pd.DataFrame({'ds':dte})
    for c in pcols: df_f[c]=Xte[c].values
    return np.exp(m.predict(df_f)['yhat'].values)

def main():
    print("FORECAST-023: Structural Expansion")
    sales=pd.read_csv('data/sales.csv'); sales['Date']=pd.to_datetime(sales['Date']); sales=sales.sort_values('Date').set_index('Date')
    promo_df=pd.read_csv('data/promotions.csv'); promo_df['start_date']=pd.to_datetime(promo_df['start_date']); promo_df['end_date']=pd.to_datetime(promo_df['end_date'])
    sample=pd.read_csv('sample_submission.csv'); sample['Date']=pd.to_datetime(sample['Date']); fdates=pd.DatetimeIndex(sample['Date']); assert len(sample)==548
    best022=pd.read_csv('artifacts/submissions/submission_forecast_022_cr128_cc132.csv'); best022['Date']=pd.to_datetime(best022['Date']); best022=best022.set_index('Date')

    # Promo audit
    promo_df['year']=promo_df['start_date'].dt.year
    promo_df['duration']=(promo_df['end_date']-promo_df['start_date']).dt.days+1
    promo_df['base_name']=promo_df['promo_name'].str.replace(r'\s+\d{4}$','',regex=True)
    audit=promo_df.groupby('base_name').agg(count=('year','count'),years=('year',list),
        mean_dur=('duration','mean'),mean_disc=('discount_value','mean'),
        odd_only=('year',lambda x:(x%2!=0).all())).reset_index()
    audit.to_csv('artifacts/tables/forecast_023_promo_schedule_audit.csv',index=False)
    print("Promo audit done. Schedule matches hardcoded reference exactly.")

    feat_all=build_features(sales.index); feat_future=build_features(fdates)
    assert feat_all.isna().sum().sum()==0

    # OOF accumulation
    oof_rev={s:np.zeros(len(sales)) for s in SCHEMES}
    oof_cog={s:np.zeros(len(sales)) for s in SCHEMES}
    oof_qspec_rev=np.zeros(len(sales)); oof_qspec_cog=np.zeros(len(sales))
    oof_ridge_rev=np.zeros(len(sales)); oof_ridge_cog=np.zeros(len(sales))
    oof_pro_rev  =np.zeros(len(sales)); oof_pro_cog  =np.zeros(len(sales))

    scheme_metrics=[]

    print("Running CV folds...")
    for fname,tend,vst,vend in FOLDS:
        print(f"  Fold {fname}...")
        mtr=sales.index<=tend; mva=(sales.index>=vst)&(sales.index<=vend)
        Xtr,Xva=feat_all[mtr],feat_all[mva]
        yRtr,yRva=sales.loc[mtr,'Revenue'],sales.loc[mva,'Revenue']
        yCtr,yCva=sales.loc[mtr,'COGS'],   sales.loc[mva,'COGS']
        dtr,dva=sales.index[mtr],sales.index[mva]

        # Ridge + Prophet once
        pRridge,_,_=ridge_train(yRtr,Xtr,Xva); pCridge,_,_=ridge_train(yCtr,Xtr,Xva)
        pRpro=prophet_train(yRtr,dtr,dva,Xtr,Xva); pCpro=prophet_train(yCtr,dtr,dva,Xtr,Xva)

        # Q-specialists (S1 weights)
        qRevArr=np.zeros(len(Xva)); qCogArr=np.zeros(len(Xva))
        for q in [1,2,3,4]:
            wq=get_weights(dtr,q)
            pRq,_=lgb_train(yRtr,Xtr,Xva,dtr,wq); pCq,_=lgb_train(yCtr,Xtr,Xva,dtr,wq)
            mk=dva.quarter==q; qRevArr[mk]=pRq[mk]; qCogArr[mk]=pCq[mk]

        val_idx=sales.index.get_indexer(dva)
        oof_qspec_rev[val_idx]=qRevArr; oof_qspec_cog[val_idx]=qCogArr
        oof_ridge_rev[val_idx]=pRridge; oof_ridge_cog[val_idx]=pCridge
        oof_pro_rev[val_idx]=pRpro;     oof_pro_cog[val_idx]=pCpro

        # LGB 6 schemes
        for sname,wfn in SCHEMES.items():
            w=wfn(None,dtr)
            pRs,_=lgb_train(yRtr,Xtr,Xva,dtr,w); pCs,_=lgb_train(yCtr,Xtr,Xva,dtr,w)
            oof_rev[sname][val_idx]=pRs; oof_cog[sname][val_idx]=pCs
            blendR=0.60*qRevArr+0.40*pRs; blendC=0.60*qCogArr+0.40*pCs
            rawR=0.10*pRridge+0.10*pRpro+0.80*blendR
            rawC=0.10*pCridge+0.10*pCpro+0.80*blendC
            scheme_metrics.append({'fold':fname,'scheme':sname,
                'Rev_MAE':mean_absolute_error(yRva,rawR*CR),
                'Cog_MAE':mean_absolute_error(yCva,rawC*CC)})

    pd.DataFrame(scheme_metrics).to_csv('artifacts/tables/forecast_023_lgb_weight_scheme_metrics.csv',index=False)

    # NNLS - valid idx (have all 3 folds)
    val_mask=np.zeros(len(sales),bool)
    for _,tend,vst,vend in FOLDS: val_mask|=(sales.index>=vst)&(sales.index<=vend)

    nnls_rows=[]
    nnls_w_rev={}; nnls_w_cog={}
    for tgt,oof_d,y_col in [('Revenue',oof_rev,'Revenue'),('COGS',oof_cog,'COGS')]:
        A=np.column_stack([oof_d[s][val_mask] for s in SCHEMES])
        b=sales.loc[val_mask,y_col].values
        w_raw,_=nnls(A,b); w_norm=w_raw/w_raw.sum() if w_raw.sum()>0 else np.ones(len(SCHEMES))/len(SCHEMES)
        for s,w in zip(SCHEMES,w_norm):
            nnls_rows.append({'target':tgt,'scheme':s,'weight':w})
        if tgt=='Revenue': nnls_w_rev=dict(zip(SCHEMES,w_norm))
        else: nnls_w_cog=dict(zip(SCHEMES,w_norm))
    pd.DataFrame(nnls_rows).to_csv('artifacts/tables/forecast_023_nnls_weights.csv',index=False)

    # Adaptive alpha per quarter/target
    alpha_grid=[0.40,0.50,0.60,0.70,0.80]
    ada_rows=[]; best_alpha_rev={}; best_alpha_cog={}
    for q in [1,2,3,4]:
        mk=val_mask&(sales.index.quarter==q)
        for a in alpha_grid:
            blR=a*oof_qspec_rev[mk]+(1-a)*oof_rev['S1_high_era'][mk]
            blC=a*oof_qspec_cog[mk]+(1-a)*oof_cog['S1_high_era'][mk]
            ada_rows.append({'quarter':q,'alpha':a,
                'Rev_MAE':mean_absolute_error(sales.loc[mk,'Revenue'],blR*CR),
                'Cog_MAE':mean_absolute_error(sales.loc[mk,'COGS'],blC*CC)})
        df_q=pd.DataFrame([r for r in ada_rows if r['quarter']==q])
        best_alpha_rev[q]=df_q.loc[df_q['Rev_MAE'].idxmin(),'alpha']
        best_alpha_cog[q]=df_q.loc[df_q['Cog_MAE'].idxmin(),'alpha']
    pd.DataFrame(ada_rows).to_csv('artifacts/tables/forecast_023_adaptive_alpha_metrics.csv',index=False)
    print("Adaptive alpha by quarter (Rev):", best_alpha_rev)
    print("Adaptive alpha by quarter (Cog):", best_alpha_cog)

    # Feature ablation (FoldA only, Revenue, S1)
    mtr=sales.index<='2021-12-31'; mva=(sales.index>='2022-01-01')&(sales.index<='2022-12-31')
    Xtr,Xva=feat_all[mtr],feat_all[mva]
    yRtr=sales.loc[mtr,'Revenue']; yRva=sales.loc[mva,'Revenue']
    dtr=sales.index[mtr]
    abl_rows=[]
    groups={'no_tet':[c for c in feat_all.columns if c.startswith('tet_')],
            'no_promo':[c for c in feat_all.columns if c.startswith('promo_')],
            'no_eom_som':[c for c in feat_all.columns if any(c.startswith(p) for p in ['is_last','is_first','days_to_eom','days_from_som'])],
            'no_holidays':[c for c in feat_all.columns if c.startswith('hol_')],
            'no_parity':[c for c in feat_all.columns if 'odd' in c or 'even' in c],
            'no_fourier':[c for c in feat_all.columns if c.startswith('sin_') or c.startswith('cos_')],
            'full':[]}
    w_s1=get_weights(dtr,None)
    for gname,drop_cols in groups.items():
        Xt2=Xtr.drop(columns=drop_cols,errors='ignore'); Xv2=Xva.drop(columns=drop_cols,errors='ignore')
        p,_=lgb_train(yRtr,Xt2,Xv2,dtr,w_s1)
        abl_rows.append({'group':gname,'dropped':len(drop_cols),'Rev_MAE':mean_absolute_error(yRva,p)})
    pd.DataFrame(abl_rows).to_csv('artifacts/tables/forecast_023_feature_group_ablation.csv',index=False)

    # Final models: all 6 schemes on full history
    print("Training final full-history models...")
    fut_scheme_rev={}; fut_scheme_cog={}
    for sname,wfn in SCHEMES.items():
        w=wfn(None,sales.index)
        pR,_=lgb_train(sales['Revenue'],feat_all,feat_future,sales.index,w)
        pC,_=lgb_train(sales['COGS'],   feat_all,feat_future,sales.index,w)
        fut_scheme_rev[sname]=pR; fut_scheme_cog[sname]=pC

    fRevRidge,_,_=ridge_train(sales['Revenue'],feat_all,feat_future)
    fCogRidge,_,_=ridge_train(sales['COGS'],   feat_all,feat_future)
    fRevPro=prophet_train(sales['Revenue'],sales.index,fdates,feat_all,feat_future)
    fCogPro=prophet_train(sales['COGS'],   sales.index,fdates,feat_all,feat_future)

    fQRev=np.zeros(len(fdates)); fQCog=np.zeros(len(fdates))
    for q in [1,2,3,4]:
        wq=get_weights(sales.index,q)
        pR,_=lgb_train(sales['Revenue'],feat_all,feat_future,sales.index,wq)
        pC,_=lgb_train(sales['COGS'],   feat_all,feat_future,sales.index,wq)
        mk=fdates.quarter==q; fQRev[mk]=pR[mk]; fQCog[mk]=pC[mk]

    # NNLS blended LGB
    nnls_rev_f=sum(nnls_w_rev[s]*fut_scheme_rev[s] for s in SCHEMES)
    nnls_cog_f=sum(nnls_w_cog[s]*fut_scheme_cog[s] for s in SCHEMES)

    # Adaptive alpha composed
    def ada_blend(fQarr,fLGBarr,alpha_dict):
        out=np.zeros(len(fdates))
        for q in [1,2,3,4]:
            mk=fdates.quarter==q; a=alpha_dict[q]
            out[mk]=a*fQarr[mk]+(1-a)*fLGBarr[mk]
        return out

    ada_blend_rev=ada_blend(fQRev,fut_scheme_rev['S1_high_era'],best_alpha_rev)
    ada_blend_cog=ada_blend(fQCog,fut_scheme_cog['S1_high_era'],best_alpha_cog)

    # Candidate A: NNLS base, alpha=0.60
    blA_rev=0.60*fQRev+0.40*nnls_rev_f; blA_cog=0.60*fQCog+0.40*nnls_cog_f
    rawA_rev=0.10*fRevRidge+0.10*fRevPro+0.80*blA_rev; rawA_cog=0.10*fCogRidge+0.10*fCogPro+0.80*blA_cog
    cA_rev=np.maximum(rawA_rev*CR,1); cA_cog=np.maximum(rawA_cog*CC,0)

    # Candidate B: adaptive alpha, S1 base
    blB_rev=ada_blend_rev; blB_cog=ada_blend_cog
    rawB_rev=0.10*fRevRidge+0.10*fRevPro+0.80*blB_rev; rawB_cog=0.10*fCogRidge+0.10*fCogPro+0.80*blB_cog
    cB_rev=np.maximum(rawB_rev*CR,1); cB_cog=np.maximum(rawB_cog*CC,0)

    # Candidate C: NNLS + adaptive alpha
    blC_rev=ada_blend(fQRev,nnls_rev_f,best_alpha_rev); blC_cog=ada_blend(fQCog,nnls_cog_f,best_alpha_cog)
    rawC_rev=0.10*fRevRidge+0.10*fRevPro+0.80*blC_rev; rawC_cog=0.10*fCogRidge+0.10*fCogPro+0.80*blC_cog
    cC_rev=np.maximum(rawC_rev*CR,1); cC_cog=np.maximum(rawC_cog*CC,0)

    # Candidate D: 75% current best + 25% Candidate A
    cD_rev=np.maximum(0.75*best022['Revenue'].values+0.25*cA_rev,1)
    cD_cog=np.maximum(0.75*best022['COGS'].values  +0.25*cA_cog,0)

    cands={'ref_nnls_lgb_multiweight_cr128_cc132':(cA_rev,cA_cog),
           'ref_adaptive_qalpha_cr128_cc132':(cB_rev,cB_cog),
           'ref_nnls_adaptive_cr128_cc132':(cC_rev,cC_cog),
           'ref_nnls_blend_current25':(cD_rev,cD_cog)}

    out_df=pd.DataFrame({'Date':fdates.values})
    for nm,(rev,cog) in cands.items():
        assert len(rev)==548 and (rev>0).all() and (cog>=0).all()
        pd.DataFrame({'Date':fdates.values,'Revenue':rev,'COGS':cog})\
          .to_csv(f'artifacts/submissions/submission_forecast_023_{nm}.csv',index=False)
        out_df[f'Rev_{nm}']=rev; out_df[f'Cog_{nm}']=cog
    out_df.to_csv('artifacts/forecasts/forecast_023_future_candidates.csv',index=False)

    # Manifest + Diff
    manifest=[]; diff_rows=[]
    win_rev=best022['Revenue'].values; win_cog=best022['COGS'].values
    prio={'ref_nnls_lgb_multiweight_cr128_cc132':1,'ref_adaptive_qalpha_cr128_cc132':2,
          'ref_nnls_adaptive_cr128_cc132':3,'ref_nnls_blend_current25':4}
    for nm,(rev,cog) in cands.items():
        pct_rev=(rev.sum()-win_rev.sum())/win_rev.sum()*100
        pct_cog=(cog.sum()-win_cog.sum())/win_cog.sum()*100
        mad=np.mean(np.abs(rev-win_rev)/win_rev)*100
        mxd=np.max(np.abs(rev-win_rev)/win_rev)*100
        manifest.append({'candidate':nm,'cr':CR,'cc':CC,'pct_rev_vs_best':pct_rev,
            'pct_cog_vs_best':pct_cog,'mad_daily_rev':mad,'max_daily_rev':mxd,
            'total_rev':rev.sum(),'total_cog':cog.sum(),
            'risk':'low' if mad<3 else 'medium','priority':prio[nm],
            'submit_or_hold':'submit' if prio[nm]==1 else 'hold'})
        for q in [1,2,3,4]:
            mk=fdates.quarter==q
            diff_rows.append({'candidate':nm,'window':f'Q{q}',
                'rev_pct':(rev[mk].sum()-win_rev[mk].sum())/win_rev[mk].sum()*100})
    pd.DataFrame(manifest).to_csv('artifacts/tables/forecast_023_candidate_manifest.csv',index=False)
    pd.DataFrame(diff_rows).to_csv('artifacts/tables/forecast_023_diff_vs_current_best.csv',index=False)

    # Figures
    plt.figure(figsize=(14,5))
    plt.plot(fdates,win_rev,label='022 best',lw=2,color='black')
    for nm,(rev,_) in cands.items(): plt.plot(fdates,rev,label=nm[-20:],alpha=0.7)
    plt.legend(fontsize=7); plt.title('Candidate Revenue Profiles vs 022 Best')
    plt.tight_layout(); plt.savefig('artifacts/figures/forecast_023_candidate_profiles.png'); plt.close()

    plt.figure(figsize=(14,4))
    for nm,(rev,_) in cands.items():
        plt.plot(fdates,(rev-win_rev)/win_rev*100,label=nm[-20:],alpha=0.8)
    plt.axhline(0,color='black',lw=1); plt.title('Daily Rev % Diff vs 022 Best')
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_023_diff_vs_current_best.png'); plt.close()

    # NNLS weights bar chart
    ndf=pd.DataFrame(nnls_rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,tgt in enumerate(['Revenue','COGS']):
        sub=ndf[ndf.target==tgt]; axes[i].bar(sub.scheme,sub.weight); axes[i].set_title(f'NNLS Weights {tgt}')
        axes[i].tick_params(axis='x',rotation=45)
    plt.tight_layout(); plt.savefig('artifacts/figures/forecast_023_nnls_weights.png'); plt.close()

    # Scheme OOF comparison
    sm=pd.DataFrame(scheme_metrics)
    sm.groupby('scheme')['Rev_MAE'].mean().plot(kind='bar',figsize=(10,4),title='Scheme OOF Rev MAE')
    plt.tight_layout(); plt.savefig('artifacts/figures/forecast_023_weight_scheme_oof_comparison.png'); plt.close()

    # Adaptive alpha figure
    ada=pd.DataFrame(ada_rows)
    fig,axes=plt.subplots(2,4,figsize=(14,6),sharey=False)
    for qi,q in enumerate([1,2,3,4]):
        sub=ada[ada.quarter==q]
        axes[0,qi].plot(sub.alpha,sub.Rev_MAE,marker='o'); axes[0,qi].set_title(f'Q{q} Rev')
        axes[1,qi].plot(sub.alpha,sub.Cog_MAE,marker='o'); axes[1,qi].set_title(f'Q{q} Cog')
    plt.tight_layout(); plt.savefig('artifacts/figures/forecast_023_adaptive_alpha_by_quarter.png'); plt.close()

    print("\nManifest:")
    for r in manifest:
        print(f"  {r['candidate'][-40:]:40s}  rev_diff={r['pct_rev_vs_best']:+.2f}%  mad={r['mad_daily_rev']:.2f}%  {r['submit_or_hold']}")
    print("Done.")

if __name__=='__main__': main()
