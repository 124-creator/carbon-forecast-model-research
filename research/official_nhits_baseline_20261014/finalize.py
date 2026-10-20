from pathlib import Path
import json,hashlib,time
R=Path(__file__).resolve().parent
exec(compile((R/'run_synthetic.py').read_text().split('results=[]')[0],str(R/'run_synthetic.py'),'exec'))
s=json.loads((R/'synthetic_results.json').read_text());m=json.loads((R/'MARKET_RESULTS.json').read_text());g=json.loads((R/'gate.json').read_text());g['market_executed']=True;(R/'gate.json').write_text(json.dumps(g,indent=2))
agg={name:float(np.mean([r['year_horizon_rw_relative_mse'] for r in m if r['model']==name])) for name in ['NHITS','MLP','DLinear']}
zero=[]
for row in s:
 if row['kind']=='no_signal' and row['model']=='NHITS':
  a=np.load(R/f"no_signal_{row['seed']}_NHITS_predictions.npz",allow_pickle=False);zero.append({'seed':row['seed'],'model_mse':float(np.mean((a['predictions']-a['truth'])**2)),'zero_forecast_mse':float(np.mean(a['truth']**2))})
summary={'market_mean_seed_loss':agg,'no_signal_zero_control':zero,'verification':json.loads((R/'verification.json').read_text()),'causality':json.loads((R/'causality_receipt.json').read_text())}
(R/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
report=f'''# 官方 NHITS 固定基线运行报告

本目录是基础模型实施，不是创新模块阶段B；没有创新声明，没有提交或推送。

## 实际版本和执行范围
- 执行官方 PyPI `neuralforecast==1.7.7`，`torch==2.3.1+cpu`，`numpy==1.26.4`，`pytorch-lightning==2.4.0`。
- 最新元数据3.3.0要求torch>=2.9.1；3.1.2要求torch>=2.4.0，均不能与既有CPU 2.3.1兼容。明确选择旧版1.7.7，不称为最新或此前main源码的逐字复现。
- 全量官方wheel本地保存，PyPI哈希与实际执行的九份核心源码哈希比对通过；Apache-2.0 LICENSE保存。没有复制N-BEATS作者代码。
- 独立项目.venv。既有CPU包和Library DLL采用本卷只读硬链接复用，没有更新父环境或Hermes环境；链接不是物理字节隔离，禁止原地编辑这些文件。
- 首次依赖解析成功但导入暴露pyarrow26/NumPy1.26不兼容，固定ray2.10、pyarrow15.0.2、pandas2.2.3、utilsforecast0.2.12后实际导入和训练通过；uv pip check通过。没有下载GPU依赖。

## 合成学习检查
纯合成趋势、周期触发的指数衰减暂态、白噪声，每类两seed（17/29），NHITS/同库MLP/DLinear，各300次更新。训练前700，测试起点700至960每20一步，完整20步输出。
- NHITS趋势相对RW MSE：{[r['ratio'] for r in s if r['model']=='NHITS' and r['kind']=='trend']}。
- NHITS暂态相对RW MSE：{[r['ratio'] for r in s if r['model']=='NHITS' and r['kind']=='transient']}。
- NHITS白噪声相对RW MSE：{[r['ratio'] for r in s if r['model']=='NHITS' and r['kind']=='no_signal']}。
- 预定学习门为趋势/暂态每seed<0.5 RW、白噪声<=1.1 RW；通过。白噪声RW不是最佳无信号预测器，额外零预测诊断为{zero}；不能据此宣称无信号最优或普遍不伤害。
- 首次暂态误写为周期正弦，原结果保存在superseded_periodic_control；改为指数衰减冲击后重跑合成，并在最终学习门通过后重新执行市场。没有将旧控制静默替换为已完成的暂态。

## 市场固定配置（非滚动强基线复现）
广州author_reported_price，仅精确950条截至2021-12-31；无缓冲逐字节读取既审计前缀，没有请求2022边界行。首459条截至2019-12-31从零训练；2020/2021同451起点。只做单次初始fit并冻结参数，不把它写成rolling40基线。
所有模型同库窗口与可用目标掩码。官方训练右侧padding保留部分成熟窗口：h1=399、h5=395、h20=380，共399输入窗口，而不是全部完整380窗口。全20步预测之后才选择1/5/20。无外生变量、无预分解、无预训练和checkpoint加载。

按2020/2021×h1/5/20平均的RW相对MSE（先算每seed loss再平均）：
- NHITS {agg['NHITS']:.9f}
- MLP {agg['MLP']:.9f}
- DLinear {agg['DLinear']:.9f}
- RW 1.0

本配置市场全部弱于RW；不追加调参抢救，不给市场增益声明。这不是整个NHITS模型族不可用的证明。
价格MSE/RMSE/MAE完整20步在MARKET_RESULTS.json；价格变化定义为预测未来原价减发行时原价。相同发行原价相减后，原价MSE与变化MSE严格同口径相等。未报告可能误导的变化百分比指标。
没有重训或导入历史MLE/强linear；父可引用历史结果但必须明确“非本次重训”及支持/刷新差异。

## 验证和证据
24主运行×300条原始batch loss，另2次NHITS因果复训×300条。每次训练state仅保存为证据，不加载。verify.py正常和-O实际通过，重算所有20步loss、市场分年分步长分数，核451计数。
官方_normalization明确把未来20位置mask设零；实际扰动这些标签后lag归一化、shift/scale完全不变。原生窗口成熟支持实测通过。合成未见后缀扰动实际两次从零复训，参数及同前缀预测逐位相同；市场2022后缀绝未读取。
单配置优化预算完成不等于最佳优化或收敛证明；保留全loss供父验证。父独立验证尚未执行，不把子验证称为父认可。

## 运行
在本绝对目录下使用`.venv/Scripts/python.exe run_synthetic.py`，学习门通过后`run_market.py`，再`verify.py`及`-O verify.py`。bootstrap.py改为固定版本CPU重装路径；该全新下载重装路径未再次执行，本次可运行环境已实际验证。
市场原始值、forecast、state等仅本地保存；不自动纳入公共提交。文件核验清单见FILE_MANIFEST.json。
'''
(R/'REPORT_ZH.md').write_text(report,encoding='utf-8')
files=[]
for p in R.rglob('*'):
 if p.is_file() and '.venv' not in p.parts and 'superseded_periodic_control' not in p.parts and p.name!='FILE_MANIFEST.json':
  b=p.read_bytes();files.append({'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(R/'FILE_MANIFEST.json').write_text(json.dumps({'count':len(files),'files':files},indent=2))
print(json.dumps({'market_seed_mean':agg,'files_verified':len(files),'no_signal_zero':zero},indent=2))
