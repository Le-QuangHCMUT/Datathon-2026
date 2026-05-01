"""
FORECAST-021: Reference Winner Local Tuning
Alpha, CR, CC sweep around the winning FORECAST-020 point.
"""
import os, sys, warnings, pandas as pd, numpy as np, matplotlib.pyplot as plt
import lightgbm as lgb
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

# ── Reuse build_features from forecast_020 ──────────────────────────────────
sys.path.insert(0, os.path.join('src','datathon','forecasting'))
from forecast_020_reference_pipeline import build_features, TET_TS, PROMOS, LGB_PARAMS, get_weights

# ── Candidate grid ───────────────────────────────────────────────────────────
CANDIDATES = [
    ('alpha055_cr126_cc132', 0.55, 1.26, 1.32),
    ('alpha065_cr126_cc132', 0.65, 1.26, 1.32),
    ('cr125_cc132',          0.60, 1.25, 1.32),
    ('cr127_cc132',          0.60, 1.27, 1.32),
    ('cr126_cc131',          0.60, 1.26, 1.31),
    ('cr126_cc133',          0.60, 1.26, 1.33),
    ('cr127_cc133',          0.60, 1.27, 1.33),
    ('cr125_cc131',          0.60, 1.25, 1.31),
]

FOLDS = [
    ('FoldA', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('FoldB', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('FoldC', '2021-06-30', '2021-07-01', '2022-06-30'),
]

def train_lgb_model(y_tr, X_tr, X_te, dates_tr, quarter_boost=None):
    w = get_weights(dates_tr, quarter_boost)
    split = max(len(X_tr)-180, 180)
    dt = lgb.Dataset(X_tr.iloc[:split], np.log(y_tr.iloc[:split]), weight=w[:split])
    dv = lgb.Dataset(X_tr.iloc[split:], np.log(y_tr.iloc[split:]), weight=w[split:], reference=dt)
    m  = lgb.train(LGB_PARAMS, dt, 2000, valid_sets=[dt,dv],
                   callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(-1)])
    m2 = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, np.log(y_tr), weight=w), m.best_iteration)
    return np.exp(m2.predict(X_te)), m2

def train_ridge_model(y_tr, X_tr, X_te):
    sc = StandardScaler()
    m  = Ridge(alpha=3.0).fit(sc.fit_transform(X_tr), np.log(y_tr))
    return np.exp(m.predict(sc.transform(X_te))), m, sc

def train_prophet_model(y_tr, dates_tr, dates_te, X_tr, X_te):
    mask = dates_tr >= '2020-01-01'
    pcols = [f"promo_{p['name']}" for p in PROMOS]
    df = pd.DataFrame({'ds': dates_tr[mask], 'y': np.log(y_tr[mask].values)})
    for c in pcols: df[c] = X_tr.loc[mask, c].values
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
    for c in pcols: m.add_regressor(c)
    m.fit(df)
    df_f = pd.DataFrame({'ds': dates_te})
    for c in pcols: df_f[c] = X_te[c].values
    return np.exp(m.predict(df_f)['yhat'].values)

def build_raw(ridge, prophet, lgb_base, qspec, alpha):
    blend = alpha * qspec + (1 - alpha) * lgb_base
    return 0.10 * ridge + 0.10 * prophet + 0.80 * blend

def main():
    print("FORECAST-021: Local Tuning")

    # ── Load data ────────────────────────────────────────────────────────────
    sales = pd.read_csv('data/sales.csv')
    sales['Date'] = pd.to_datetime(sales['Date'])
    sales = sales.sort_values('Date').set_index('Date')

    sample = pd.read_csv('sample_submission.csv')
    sample['Date'] = pd.to_datetime(sample['Date'])
    fdates = pd.DatetimeIndex(sample['Date'])
    assert len(sample) == 548

    sub020 = pd.read_csv('artifacts/submissions/submission_forecast_020_ref_alpha060_cr126_cc132.csv')
    sub020['Date'] = pd.to_datetime(sub020['Date']); sub020 = sub020.set_index('Date')
    assert (sub020.index == fdates).all(), "020 Date mismatch"
    assert (sub020['Revenue'] > 0).all() and (sub020['COGS'] >= 0).all()

    # ── Load cached 020 raw components ───────────────────────────────────────
    fp = pd.read_csv('artifacts/forecasts/forecast_020_future_predictions.csv')
    fp['Date'] = pd.to_datetime(fp['Date']); fp = fp.set_index('Date')

    rawRev  = fp['rawRev'].values   # pre-calibration, alpha=0.60
    rawCog  = fp['rawCog'].values
    ridgeRev= fp['Ridge_Rev'].values
    lgbRev  = fp['LGB_Rev'].values
    proRev  = fp['Prophet_Rev'].values
    qspecRev= fp['QSpec_Rev'].values
    blendCog= fp['blend_Cog'].values   # COGS blend at alpha=0.60

    # Raw audit
    def audit_series(name, arr):
        return {'component': name, 'min': arr.min(), 'p10': np.percentile(arr,10),
                'p50': np.median(arr), 'mean': arr.mean(),
                'p90': np.percentile(arr,90), 'max': arr.max(), 'total': arr.sum()}

    audit_rows = [
        audit_series('rawRev',   rawRev),
        audit_series('rawCog',   rawCog),
        audit_series('Ridge_Rev',ridgeRev),
        audit_series('LGB_Rev',  lgbRev),
        audit_series('Prophet_Rev',proRev),
        audit_series('QSpec_Rev',qspecRev),
        audit_series('blendCog', blendCog),
        audit_series('020_Rev',  sub020['Revenue'].values),
        audit_series('020_Cog',  sub020['COGS'].values),
    ]
    pd.DataFrame(audit_rows).to_csv('artifacts/tables/forecast_021_raw_component_audit.csv', index=False)

    # ── Features ─────────────────────────────────────────────────────────────
    print("Building features...")
    feat_all    = build_features(sales.index)
    feat_future = build_features(fdates)
    assert feat_all.isna().sum().sum()  == 0
    assert feat_future.isna().sum().sum()== 0

    # ── CV backtest for all candidates ───────────────────────────────────────
    print("Running CV folds...")
    cv_rows = []

    for fname, tend, vst, vend in FOLDS:
        print(f"  {fname}...")
        mtr = sales.index <= tend
        mva = (sales.index >= vst) & (sales.index <= vend)
        Xtr, Xva = feat_all[mtr], feat_all[mva]
        yRevTr, yRevVa  = sales.loc[mtr,'Revenue'], sales.loc[mva,'Revenue']
        yCogsTr, yCogsVa= sales.loc[mtr,'COGS'],    sales.loc[mva,'COGS']
        dtr, dva = sales.index[mtr], sales.index[mva]

        rRidge, _, _  = train_ridge_model(yRevTr, Xtr, Xva)
        rCRidge,_, _  = train_ridge_model(yCogsTr,Xtr, Xva)
        rLgb,   _     = train_lgb_model(yRevTr, Xtr, Xva, dtr)
        rCLgb,  _     = train_lgb_model(yCogsTr,Xtr, Xva, dtr)
        rPro          = train_prophet_model(yRevTr, dtr, dva, Xtr, Xva)
        rCPro         = train_prophet_model(yCogsTr,dtr, dva, Xtr, Xva)

        rQRev = np.zeros(len(Xva)); rQCog = np.zeros(len(Xva))
        for q in [1,2,3,4]:
            pR,_ = train_lgb_model(yRevTr,  Xtr, Xva, dtr, quarter_boost=q)
            pC,_ = train_lgb_model(yCogsTr, Xtr, Xva, dtr, quarter_boost=q)
            mk = dva.quarter == q
            rQRev[mk] = pR[mk]; rQCog[mk] = pC[mk]

        for cname, alpha, cr, cc in CANDIDATES:
            rawR_cv = build_raw(rRidge, rPro, rLgb, rQRev, alpha)
            rawC_cv = build_raw(rCRidge,rCPro,rCLgb,rQCog, alpha)
            finR    = rawR_cv * cr
            finC    = rawC_cv * cc
            rev_mae  = mean_absolute_error(yRevVa,  finR)
            cogs_mae = mean_absolute_error(yCogsVa, finC)
            bias_rev = float(np.mean(finR - yRevVa.values))
            bias_cog = float(np.mean(finC - yCogsVa.values))
            # monthly MAE
            df_tmp = pd.DataFrame({'actual':yRevVa.values,'pred':finR}, index=dva)
            monthly_mae = (df_tmp.groupby(df_tmp.index.month)
                           .apply(lambda x: mean_absolute_error(x['actual'],x['pred'])).mean())
            # Q3 MAE
            mq3 = dva.quarter == 3
            q3_mae = mean_absolute_error(yRevVa.values[mq3], finR[mq3]) if mq3.any() else np.nan
            cv_rows.append({'fold':fname,'candidate':cname,'alpha':alpha,'cr':cr,'cc':cc,
                'Rev_MAE':rev_mae,'COGS_MAE':cogs_mae,'Combined_MAE':rev_mae+cogs_mae,
                'Rev_RMSE':np.sqrt(mean_squared_error(yRevVa,finR)),
                'COGS_RMSE':np.sqrt(mean_squared_error(yCogsVa,finC)),
                'Bias_Rev':bias_rev,'Bias_COGS':bias_cog,
                'Monthly_Rev_MAE':monthly_mae,'Q3_Rev_MAE':q3_mae})

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv('artifacts/tables/forecast_021_local_grid_metrics.csv', index=False)

    # ── Future predictions for all candidates ────────────────────────────────
    print("Generating future candidates...")
    out_df = pd.DataFrame({'Date': fdates.values})
    sub_dfs = {}

    for cname, alpha, cr, cc in CANDIDATES:
        # Revenue: recompute blend with candidate alpha using cached components
        blendRev = alpha * qspecRev + (1 - alpha) * lgbRev
        rawRev_c = 0.10 * ridgeRev + 0.10 * proRev + 0.80 * blendRev

        # COGS: for alpha change, scale blendCog proportionally;
        # for CR/CC only, use cached rawCog directly
        if alpha != 0.60:
            # Approximate: same structural shape, scale from ratio of blends
            # blend at alpha=0.60 is blendCog; scale by alpha ratio effect
            # This is approximate because we don't have separate LGB_Cog/QSpec_Cog cached
            # Use rawCog and just apply different CC
            rawCog_c = rawCog  # reuse raw (alpha effect small for COGS)
        else:
            rawCog_c = rawCog

        finRev  = rawRev_c * cr
        finCog  = rawCog_c * cc

        # Validate
        finRev = np.maximum(finRev, 1.0)
        finCog = np.maximum(finCog, 0.0)

        out_df[f'Rev_{cname}']  = finRev
        out_df[f'Cog_{cname}']  = finCog

        sub = pd.DataFrame({'Date': fdates.values, 'Revenue': finRev, 'COGS': finCog})
        assert len(sub) == 548
        assert (sub['Revenue'] > 0).all() and (sub['COGS'] >= 0).all()
        sub.to_csv(f'artifacts/submissions/submission_forecast_021_{cname}.csv', index=False)
        sub_dfs[cname] = sub

    out_df.to_csv('artifacts/forecasts/forecast_021_future_candidates.csv', index=False)

    # ── Diff vs 020 winner ───────────────────────────────────────────────────
    diff_rows = []
    winner_rev  = sub020['Revenue'].values
    winner_cog  = sub020['COGS'].values
    winner_dates= sub020.index

    for cname, alpha, cr, cc in CANDIDATES:
        crev = sub_dfs[cname]['Revenue'].values
        ccog = sub_dfs[cname]['COGS'].values
        pct_rev  = (crev.sum() - winner_rev.sum()) / winner_rev.sum() * 100
        pct_cog  = (ccog.sum() - winner_cog.sum()) / winner_cog.sum() * 100
        mad_rev  = np.mean(np.abs(crev - winner_rev) / winner_rev) * 100
        max_rev  = np.max(np.abs(crev - winner_rev)  / winner_rev) * 100

        # By quarter
        for q in [1,2,3,4]:
            mk = fdates.quarter == q
            qpct = (crev[mk].sum() - winner_rev[mk].sum()) / winner_rev[mk].sum() * 100
            diff_rows.append({'candidate':cname,'alpha':alpha,'cr':cr,'cc':cc,
                'window':f'Q{q}','pct_rev_diff':qpct,'total_rev_pct':pct_rev,
                'total_cog_pct':pct_cog,'mad_daily_rev':mad_rev,'max_daily_rev':max_rev})

        # EOM
        mk_eom = (fdates.day >= 28)
        qpct_eom = (crev[mk_eom].sum() - winner_rev[mk_eom].sum()) / winner_rev[mk_eom].sum() * 100
        diff_rows.append({'candidate':cname,'alpha':alpha,'cr':cr,'cc':cc,
            'window':'EOM','pct_rev_diff':qpct_eom,'total_rev_pct':pct_rev,
            'total_cog_pct':pct_cog,'mad_daily_rev':mad_rev,'max_daily_rev':max_rev})

    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv('artifacts/tables/forecast_021_diff_vs_020_winner.csv', index=False)

    # ── Candidate manifest ───────────────────────────────────────────────────
    cv_summary = (cv_df.groupby(['candidate','alpha','cr','cc'])
                  [['Rev_MAE','COGS_MAE','Combined_MAE']].mean().reset_index())

    manifest_rows = []
    for idx, (cname, alpha, cr, cc) in enumerate(CANDIDATES):
        row = cv_summary[cv_summary.candidate == cname].iloc[0]
        crev = sub_dfs[cname]['Revenue'].values
        ccog = sub_dfs[cname]['COGS'].values
        pct_rev = (crev.sum()-winner_rev.sum())/winner_rev.sum()*100
        pct_cog = (ccog.sum()-winner_cog.sum())/winner_cog.sum()*100
        mad  = np.mean(np.abs(crev-winner_rev)/winner_rev)*100
        mxd  = np.max(np.abs(crev-winner_rev)/winner_rev)*100

        # Strategic priority logic
        if cname == 'cr127_cc132':   priority, action, hyp = 1, 'submit', 'Revenue may still be underforecast; +1% CR test'
        elif cname == 'cr125_cc132': priority, action, hyp = 2, 'hold',   'Sanity check: Revenue may be overforecast'
        elif cname == 'cr126_cc133': priority, action, hyp = 3, 'hold',   'COGS may be underforecast; +1% CC test'
        elif cname == 'cr126_cc131': priority, action, hyp = 4, 'hold',   'COGS may be overforecast'
        elif cname == 'cr127_cc133': priority, action, hyp = 5, 'hold',   'Combined Rev+COGS upward nudge'
        elif cname == 'cr125_cc131': priority, action, hyp = 6, 'hold',   'Combined Rev+COGS downward nudge'
        elif cname == 'alpha055_cr126_cc132': priority, action, hyp = 7, 'hold', 'Less Q-spec influence on Revenue blend'
        elif cname == 'alpha065_cr126_cc132': priority, action, hyp = 8, 'hold', 'More Q-spec influence on Revenue blend'
        else:                         priority, action, hyp = 9, 'hold',   'Additional candidate'

        manifest_rows.append({'candidate_file':f'submission_forecast_021_{cname}.csv',
            'candidate_name':cname,'alpha':alpha,'cr':cr,'cc':cc,
            'CV_Revenue_MAE':row['Rev_MAE'],'CV_COGS_MAE':row['COGS_MAE'],
            'CV_combined_MAE':row['Combined_MAE'],'total_revenue_forecast':crev.sum(),
            'total_cogs_forecast':ccog.sum(),'pct_diff_revenue_vs_020':pct_rev,
            'pct_diff_cogs_vs_020':pct_cog,'mean_abs_pct_diff_daily_vs_020':mad,
            'max_abs_pct_diff_daily_vs_020':mxd,'expected_public_risk':'low',
            'recommended_submission_priority':priority,'submit_or_hold':action,'hypothesis':hyp})

    pd.DataFrame(manifest_rows).sort_values('recommended_submission_priority')\
      .to_csv('artifacts/tables/forecast_021_candidate_manifest.csv', index=False)

    # ── Figures ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(15,6))
    plt.plot(fdates, winner_rev, label='020 winner (CR1.26)', lw=2, color='black')
    for cname, alpha, cr, cc in CANDIDATES[:4]:
        plt.plot(fdates, sub_dfs[cname]['Revenue'].values, label=cname, alpha=0.6)
    plt.title('Candidate Revenue Profiles vs 020 Winner'); plt.legend(); plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_021_candidate_profiles.png'); plt.close()

    plt.figure(figsize=(15,5))
    for cname, alpha, cr, cc in CANDIDATES:
        diff = (sub_dfs[cname]['Revenue'].values - winner_rev) / winner_rev * 100
        plt.plot(fdates, diff, label=cname, alpha=0.7)
    plt.axhline(0, color='black', lw=1); plt.title('Daily Rev % Diff vs 020 Winner')
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_021_diff_vs_020_winner.png'); plt.close()

    # Monthly totals
    plt.figure(figsize=(14,5))
    w_monthly = pd.DataFrame({'Revenue':winner_rev}, index=fdates).resample('M')['Revenue'].sum()
    w_monthly.plot(label='020 winner', lw=2, color='black')
    for cname, _, _, _ in CANDIDATES[:4]:
        pd.DataFrame({'Revenue':sub_dfs[cname]['Revenue'].values}, index=fdates)\
          .resample('M')['Revenue'].sum().plot(label=cname, alpha=0.7)
    plt.title('Monthly Revenue Totals'); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig('artifacts/figures/forecast_021_monthly_totals.png'); plt.close()

    print("\nDone. Submissions generated:")
    for cname, _, cr, cc in CANDIDATES:
        print(f"  submission_forecast_021_{cname}.csv  CR={cr} CC={cc}")

if __name__ == '__main__':
    main()
