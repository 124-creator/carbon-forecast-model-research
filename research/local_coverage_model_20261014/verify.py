import json,csv
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
R=json.loads((O/'RESULTS.json').read_text());L=json.loads((O/'TRAINING_LOGS.json').read_text());C=json.loads((O/'CAUSAL_AUDIT.json').read_text());S=json.loads((O/'SYNTHETIC.json').read_text());receipt=json.loads((O/'RUN_RECEIPT.json').read_text())
def check(c,m):
 if not c:raise ValueError(m)
a=np.load(O/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);errors=[];flat=[]
for r in R:
 vals=[]
 for row in r['by_year']:
  mask=a['years']==row['year'];mse=np.mean((a[r['name']][mask]-a['truth'][mask])**2,0);den=np.mean(a['truth'][mask]**2,0);rel=mse/den;errors.extend(abs(rel-np.array(row['relative_mse_h'])));vals.extend(rel)
  for j,h in enumerate([1,5,20]):flat.append(dict(model=r['name'],year=row['year'],h=h,n=int(mask.sum()),mse=float(mse[j]),relative_mse=float(rel[j])))
 errors.append(abs(np.mean(vals)-r['score']))
check(max(errors)<1e-12,'score replay')
market=[r for r in L if r['label']=='market'];groups={}
for r in market:
 check(r['max_target']<r['fit_end'],'outer maturity');check(r['inner_max_target']<r['validation_first_origin'],'inner purge');groups.setdefault((r['fit_end'],r['h']),set()).add((r['min_origin'],r['max_origin'],r['n']))
check(all(len(v)==1 for v in groups.values()),'same per-h support');check(max(r['augmented_prediction_difference'] for r in L)<1e-6,'equivalence');check(all(max([v for k,v in r.items() if 'difference' in k] or [0])<1e-8 for r in C),'causality')
with (O/'ALL_YEAR_H_ABLATIONS.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
verification=dict(score_max_error=float(max(errors)),score_cells=len(flat),training_records=len(L),market_fit_h_groups=len(groups),matched_support=True,purged=True,causal=True,max_augmented_difference=max(r['augmented_prediction_difference'] for r in L),seconds=receipt['seconds'],optimized=not __debug__)
(O/('VERIFY_OPTIMIZED.json' if not __debug__ else 'VERIFY_NORMAL.json')).write_text(json.dumps(verification,indent=2));print(json.dumps(verification));print(json.dumps(S,indent=2))
lines=['# 相同强信息接口下的局部覆盖诊断','','本实现没有方法创新资格：固定训练状态后，覆盖质量对角项加二阶差分平滑算子严格是数据依赖 generalized ridge / P-spline。增广最小二乘数值核对已完成。不把覆盖质量改称有效样本量，也不声称查新完成。','', '## 定义与接口','每h输入只有当时签发的MLE预测、60点价格状态和20/120波动比，沿用state/history，不使用当前参数回填历史预测。目标为原价增量减去历史签发MLE；所有方法学习同一残差，再加回当前签发MLE。这是校正诊断，不是local_basis_review所要求的全新联合直接预测模型。','每40点刷新。h分别使用最多480个成熟起点，s+h<fit_end；各方法在同一h完全相同。内验证最后120个起点，训练标签严格早于第一个验证起点。节点为该h该次训练内分位数.1/.3/.5/.7/.9，次数1。内验证单独估计节点/覆盖，不复用外训练估计。','令L=[1,z]，B为局部线性B样条，N=(B-L pinv(L)B)/std(B-L pinv(L)B)。覆盖m_k=mean_s B_sk²，不加时间权重，不是ESS。v_k=1/max(m_k,1/n)，再使mean(v)=1；C=diag(v)+D2ᵀD2，再使mean(diag C)=1。目标 ||y-[L,N]β||²+nαβᵀPβ；线性和普通样条P为非截距单位阵；候选非线性块为C。','覆盖与平滑是具体实现区别，但仍完全属于常规自适应广义岭；不是独立的新函数类。constant与普通样条严格相同；permuted反转整个非线性算子的位置，是位置负对照，不能单独识别质量估计和平滑两个部件。未追加看到结果后的正则/节点搜索。','','## 全部原价开发成绩','950行广州作者报告价格，451个同年完整评分起点，2020/2021都是反复开发段，无外生数据/分解。未请求2022边界记录或全文件哈希。固定α=.1、10，保留所有结果与各h内验证选择，不采用市场事后最佳当独立证据。','', '|方法|等权相对RW MSE|','|---|---|']
for r in R:lines.append('|'+r['name']+'|'+format(r['score'],'.9f')+'|')
lines+=['','全分年h原价MSE、相对MSE和数量见ALL_YEAR_H_ABLATIONS.csv。','', '## 合成与结论','先执行两个固定种子的局部状态覆盖漂移非线性正控制与无信号控制，后执行市场。合成原始状态仅模拟相同三维接口，并非生成价格后再滤波得到真实MLE；合成使用h20成熟支持，不能据此声称各h机制成立。全部两档成绩保留于SYNTHETIC.json，未用控制结果追加参数。','覆盖候选内验证成绩比普通样条更差，也不胜RW/MLE；固定两档亦无稳定优势。普通样条本身也未超过MLE，不能声称简单样条已解决任务。否决此实现的新贡献，并结束此实现；整体模型研究继续，但不能在此追加配置挽救成绩。','原state线性最佳约.942705采用所有系数包括截距的零中心惩罚，本实现截距不罚；因此当前线性分数不是那个历史校正的复现，不能把不一致归因于输入信息。','', '## 核查与局限','普通/优化模式独立评分、增广精确解、每h相同支持、内验证清除跨界目标、签发特征前缀重放、未来价格扰动后重构状态/目标/模型均已核查。当前块内预测使用当前因果状态，参数保持固定。两档及所有日志保留；本地价格相关NPZ由.gitignore排除。','候选与平滑的单独消融未预注册，故不追加；只有覆盖算子整体/常数/位置负对照，不足以把任何改变量归因于单独覆盖估计。','计算用时 '+str(receipt['seconds'])+' 秒，预算600秒。没有提交或推送。']
(O/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
