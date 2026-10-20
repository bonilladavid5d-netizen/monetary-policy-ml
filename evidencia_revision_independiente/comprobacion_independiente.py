"""Recalcula métricas desde las tablas y el objetivo real desde los CSV originales.
No importa funciones de evaluación del proyecto ni modifica la fuente."""
from pathlib import Path
import json, hashlib, argparse
import pandas as pd
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'source')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
R=args.source.resolve()
O=args.output.resolve();O.mkdir(parents=True,exist_ok=True)
res={}
hashes=[]
for line in (R/'MANIFEST.sha256').read_text().splitlines():
    if line.startswith('#') or not line.strip(): continue
    digest,path=line.split(maxsplit=1)
    hashes.append(hashlib.sha256((R/path).read_bytes()).hexdigest()==digest)
res['manifest']={'entries':len(hashes),'matching':sum(hashes)}

# Objetivo: implementación independiente directamente desde dos insumos FRED.
raw={}
for name in ['CPIAUCSL','UNRATE']:
    d=pd.read_csv(R/f'data/raw/{name}.csv',parse_dates=['observation_date']).set_index('observation_date')
    raw[name]=d[name]
pi=100*(raw['CPIAUCSL']/raw['CPIAUCSL'].shift(1)-1)
u=raw['UNRATE']
y=pd.Series(np.nan,index=pi.index)
for i in range(len(y)-12):
    y.iloc[i]=pi.iloc[i+1:i+13].sum()+u.iloc[i+1:i+13].mean()
p=pd.read_csv(R/'tablas/real_predicciones.csv',parse_dates=['date'])
res['real_target_max_abs_diff']=float(np.max(np.abs(p.Y.to_numpy()-y.reindex(p.date).to_numpy())))
rep=pd.read_csv(R/'tablas/real_metricas.csv')
errs=[]
for row in rep.itertuples():
    sub=p[p.variante==row.variante]
    sub=sub[sub.bloque.str.startswith('fold')] if row.bloque=='folds_desarrollo_agregado' else sub[sub.bloque==row.bloque]
    e=sub[f'pred_{row.modelo}']-sub.Y
    errs.extend([abs(np.sqrt(np.mean(e**2))-row.rmse),abs(np.mean(abs(e))-row.mae)])
res['real_metrics_max_abs_diff']=float(max(errs))
res['real_test_metrics']=rep[(rep.bloque=='test')][['variante','modelo','n','rmse','mae']].to_dict('records')

# q independiente, usando las ecuaciones explícitas del DGP original.
s=pd.read_csv(R/'tablas/sintetico_predicciones.csv')
Q=[]
for delta in [0,.5,-.5]:
    gap=s.fedfunds+delta-2.5
    mpi=.75*(s.inflation-2)+2-.35*(1+.5*np.tanh(s.inflation-2))*gap+.25*s.activity
    mu=.85*(s.unemployment-4.5)+4.5-.30*s.activity+.20*gap
    Q.append((mpi-2)**2+.30**2+(mu-4.5)**2+.20**2+2*delta**2)
q=np.column_stack(Q)
res['synthetic_q_max_abs_diff']=float(np.max(np.abs(q-s[['eval_q_T0','eval_q_T1','eval_q_T2']].to_numpy())))
M=pd.read_csv(R/'tablas/sintetico_por_semilla.csv')
er=[]
recalc=[]
for row in M.itertuples():
    ix=np.flatnonzero(s.semilla.to_numpy()==row.semilla)
    ss=s.iloc[ix];qq=q[ix]
    if row.politica in ['ridge','rf']:
        action=ss[f'{row.politica}_accion'].str[-1].astype(int).to_numpy()
    elif row.politica.startswith('siempre_'):
        action=np.repeat(int(row.politica[-1]),len(ix))
    else:
        action=qq.argmin(axis=1)
    L=qq[np.arange(len(ix)),action]
    regret=float(np.mean(L-qq.min(axis=1)))
    er.extend([abs(regret-row.regret_esperado_medio),abs(float(L.mean())-row.perdida_esperada_media)])
    recalc.append({'seed':row.semilla,'policy':row.politica,'regret':regret,'loss':float(L.mean())})
res['synthetic_metrics_max_abs_diff']=float(max(er))
G=pd.DataFrame(recalc).groupby('policy')[['regret','loss']].mean()
res['synthetic_averages']=G.to_dict('index')
res['rf_vs_T0']={'regret_reduction':float(1-G.loc['rf','regret']/G.loc['siempre_T0','regret']),
    'total_loss_reduction':float(1-G.loc['rf','loss']/G.loc['siempre_T0','loss'])}
S=s[s.rf_empate.str.contains('|',regex=False)]
res['rf_ties']={'count':len(S),'flat':int((s.rf_empate=='T0|T1|T2').sum()),'resolved_T0':int((S.rf_accion=='T0').sum())}
res['synthetic_tuning_boundary']={'last_dev_index':479,'last_dev_label_time':480,'test_first':480,
    'interpretation':'Asimetría de la purga conservadora, no observación del choque futuro de la decisión480.'}
(O/'comprobacion_independiente.json').write_text(json.dumps(res,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(res,indent=2,ensure_ascii=False))
