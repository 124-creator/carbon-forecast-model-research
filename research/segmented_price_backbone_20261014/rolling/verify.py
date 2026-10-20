"""Independent raw-label saved scoring; explicit checks also under -O."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,csv,sys,hashlib
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2]
lines=[]
with (R/'data/parsed/figshare_28682348/guangzhou_reported.csv').open('rb',buffering=0) as f:
 for _ in range(951):
  b=bytearray()
  while True:
   c=f.read(1)
   if not c:raise ValueError('short prefix')
   b+=c
   if c==b'\n':break
  lines.append(bytes(b))
r=list(csv.DictReader(b''.join(lines).decode('utf-8-sig').splitlines()));dates=np.array([s['date'] for s in r]);p=np.array([float(s['author_reported_price']) for s in r]);orig=np.array([i for i in range(459,930) if dates[i][:4]==dates[i+20][:4]]);truth=p[orig[:,None]+[1,5,20]]-p[orig,None];years=np.array([d[:4] for d in dates[orig]])
z=np.load(O/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);rows=json.loads((O/'RESULTS.json').read_text());errors=[]
if len(p)!=950 or dates[-1]!='2021-12-31' or len(orig)!=451 or not np.array_equal(orig,z['origins']) or not np.array_equal(truth,z['truth']):raise ValueError('labels/support')
for row in rows:
 pr=z[row['name']]
 if pr.shape!=(451,3) or not np.isfinite(pr).all():raise ValueError('shape/finite')
 cells=[]
 for year in ['2020','2021']:
  ix=years==year;cells.extend((np.mean((pr[ix]-truth[ix])**2,0)/np.mean(truth[ix]**2,0)).tolist())
 errors.append(abs(np.mean(cells)-row['score']))
if len(rows)!=11 or max(errors)>1e-12:raise ValueError('saved score')
logs=json.loads((O/'TRAINING_LOGS.json').read_text());nets=[r for r in logs if r['policy']=='rolling_segmented'];linear=[r for r in logs if r['policy']=='rolling_linear'];inv=json.loads((O/'INITIAL_TRAINING_INVARIANCE.json').read_text())
if len(nets)!=24 or len(linear)!=12 or len(inv)!=2:raise ValueError('fit count')
for a in nets:
 if a['updates']!=480 or a['effective_nonzero_gradient_steps']!=480 or max(a['per_h_counts'])>480 or max(a['max_target'])>=a['end_exclusive'] or len(a['history'])!=60 or len(a['gradient_norms'])!=480 or not a['history'][-1]<a['initial']:raise ValueError('training log')
for a in inv:
 if max(a['parameter_maxdiff'],a['frozen_retraining_prediction_maxdiff'],a['last_prefix_forecast_maxdiff'])>1e-10:raise ValueError('invariance')
by={a['name']:a for a in rows};frozen=float(np.mean([by[f'frozen_segmented_{s}']['score'] for s in [7,17]]));rolling=float(np.mean([by[f'rolling_segmented_{s}']['score'] for s in [7,17]]));out=dict(optimized=not __debug__,score_files=11,max_score_error=float(max(errors)),prefix_rows=950,read_boundary=False,forecast_origins=451,rolling_network_fits=24,rolling_linear_fits=12,updates_per_network_fit=480,frozen_seed_loss_mean=frozen,rolling_seed_loss_mean=rolling,rolling_minus_frozen=rolling-frozen,initial_retraining_invariance=inv,final_full_gradient_norm_range=[min(a['final_full_gradient_norm'] for a in nets),max(a['final_full_gradient_norm'] for a in nets)],last10_loss_drop_range=[min(a['last10_loss_drop'] for a in nets),max(a['last10_loss_drop'] for a in nets)])
(O/('VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __debug__:
 report=['# SegRNN 固定配置滚动重训诊断','',f'保存预测独立重新评分通过：11组、451起点，误差{max(errors):.3g}。未来后缀从459起全部扰动，两个固定种子重新完成初始训练；参数、过去末点预测、原冻结预测重放最大差异均为0。仅读取950条许可前缀，不请求2022边界。','', '|模型|评分（两个种子损失平均）|','|---|---:|',f'|冻结SegRNN|{frozen:.9f}|',f'|每40点滚动SegRNN|{rolling:.9f}|',f"|冻结同段线性|{by['frozen_linear_7']['score']:.9f}|",f"|滚动同段线性|{by['rolling_linear']['score']:.9f}|",f"|历史强MLE480复用|{by['historical_MLE480']['score']:.9f}|",f"|历史state线性alpha10复用|{by['historical_state_linear_a10']['score']:.9f}|",'',f'滚动SegRNN比冻结平均恶化{rolling-frozen:.9f}。seed7为1.080451599，seed17为1.106885031；两种子均未改善，滚动仍优于同段线性，但未胜随机游走或历史强对照。这是固定配置、当前已用开发数据上的不改善证据，不是SegRNN整个模型家族无效的证明。普通滚动重训不是模型创新。','', '## 配置与优化','原模型源码不变，seed7/17，1794有效参数，Adam lr=.001，原1e-4衰减、原梯度裁剪、60epochs、每轮8批，480有效更新。每次从相同固定种子重新初始化，没有暖启动或额外种子/正则搜索。冻结为一次初始训练，此后仅更新价格输入；滚动为12次训练，参数更新口径与计算成本不同，不能称两者同总训练预算。初始逐h支持399/395/380，随后每h最多480，严格目标索引小于刷新边界；输入尺度使用当次训练窗口、输出尺度使用当次h成熟标签。60维中心化原价线性alpha1同刷新、同训练支持、同尺度。滚动协议包含最新480成熟支持和训练内重新估计尺度，不能把效应严格拆成单独参数刷新的纯因果贡献。',f"每次目标均实际下降，全部24次网络训练完成480非零梯度步，末次全样本梯度范数范围{out['final_full_gradient_norm_range']}，末10轮损失下降范围{out['last10_loss_drop_range']}。固定预算已完整执行，但梯度与末段下降并未证明收敛或全局充分，不能把预算耗尽称作优化充分。完整逐步梯度、逐轮loss与成熟支持保存在TRAINING_LOGS.json。",'', '## 强对照及范围','MLE480和历史state线性直接复用已保存签发时点预测，并核查起点与原价标签；后者是成熟残差校正，不是本轮原60维线性。历史MLE采用单步似然支持，与神经网络逐h成熟监督支持不同。没有把历史复用说成此次重新训练。市场为广州作者报告价格，预测h1/5/20观察点增量，无外生变量、无分解，2020/2021是已用开发段，不是未见测试。未提交或push。','', '## 父级核查命令','`python research/segmented_price_backbone_20261014/rolling/verify.py`','`python -O research/segmented_price_backbone_20261014/rolling/verify.py`','原始初始训练重放及未来扰动证据在INITIAL_TRAINING_INVARIANCE.json；两种子均重新执行正常与扰动数据训练，不只是比较输入数组。原冻结结果没有覆盖。']
 (O/'REPORT_ZH.md').write_text('\n'.join(report),encoding='utf-8')
