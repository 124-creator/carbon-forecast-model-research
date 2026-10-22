# 可复现公开代码筛查：等长因果协议的移植候选

**8个仓库均以GitHub API HTTP200重核公开状态并固定commit；指定旧3条已重核，另增5个B类候选。建议先移植arch的HARX接口、husaam的log-HAR回归核、以及ayk中隔离出来的树估计器。第三项不包括作者有因果问题的实验runner。**本轮未执行作者代码、安装、克隆或训练，没有复现成绩。

每条有metadata、default branch、未截断recursive tree和固定SHA的contents证据；不能从README推断可跑。API size单位KB，不是下载量；时间均UTC。完整SHA、API200链接、blob SHA、数据/requirements路径和短源码片段见[JSON](<E:/Agent员工/碳价格预测研究/独立数据归档/carbon-forecast-model-research/research/neural_model_restart_20261021/REPRODUCIBLE_CODE_INVENTORY.json>)。

## 按可移植性排序

| 排名/仓库 | 类别 | 主要移植对象 | 工作量 |
|---|---|---|---|
| 1. [bashtage/arch](https://github.com/bashtage/arch) | B | HARX .fit/.forecast API；GARCH仅在允许该信息集时另列基准 | S |
| 2. [husaam-atq/volatility-risk-forecasting-platform](https://github.com/husaam-atq/volatility-risk-forecasting-platform) | B | har_models的log-Ridge/Huber回归核，不移植数据库/仪表盘 | M |
| 3. [ayk5511/volatility-forecasting](https://github.com/ayk5511/volatility-forecasting) | B | 只隔离LightGBM/XGBoost或HAR估计器，接我们的driver；原runner修复为L | M（估计器）；原runner L |
| 4. [deep-hedger-Peng/HAR-RV](https://github.com/deep-hedger-Peng/HAR-RV) | B | 只抽HAR滞后与OLS，不沿用采集/缩放流程 | M |
| 5. [Aurelien7877/VolaTTM-Cross-Asset-Realized-Volatility-Forecasting](https://github.com/Aurelien7877/VolaTTM-Cross-Asset-Realized-Volatility-Forecasting) | B | 主TTM为L；仅隔离rolling_har_predictions可为M | L（TTM）；HAR子模块M |
| 6. [weibaozhong/BMambaXer](https://github.com/weibaozhong/BMambaXer) | A | BMambaXer模型及custom数据路径 | L |
| 7. [StriveOrange/TGSINFO-VMD-NEQQLattice](https://github.com/StriveOrange/TGSINFO-VMD-NEQQLattice) | A | MATLAB分解已读；NEQQLattice预测notebook仅静态核实 | L |
| 8. [maybe-collab/carbon-price-forecasting-system](https://github.com/maybe-collab/carbon-price-forecasting-system) | A | ZIP内mian.m、BPNN.m、SVR.m已读 | L |

S/M/L是静态适配工作量，不表示运行通过。新增5个均是RV/HAR或波动率基准类B候选；范围A只覆盖三条已指定的碳价仓库。新增是相对明确给出的旧三条；未声称穷尽所有历史搜索日志。

## 当前API元数据

| full_name | branch | size KB | API language | API license | stars | pushed_at UTC |
|---|---|---:|---|---|---:|---|
| [bashtage/arch](https://github.com/bashtage/arch) | main | 484,432 | Python | NOASSERTION | 1603 | 2026-10-21T17:36:35Z |
| [husaam-atq/volatility-risk-forecasting-platform](https://github.com/husaam-atq/volatility-risk-forecasting-platform) | main | 94,520 | Python | MIT | 0 | 2026-05-22T16:53:38Z |
| [ayk5511/volatility-forecasting](https://github.com/ayk5511/volatility-forecasting) | main | 2,859 | TeX | NOASSERTION | 12 | 2026-05-04T02:05:55Z |
| [deep-hedger-Peng/HAR-RV](https://github.com/deep-hedger-Peng/HAR-RV) | master | 198 | null | null | 28 | 2020-06-22T10:35:07Z |
| [Aurelien7877/VolaTTM-Cross-Asset-Realized-Volatility-Forecasting](https://github.com/Aurelien7877/VolaTTM-Cross-Asset-Realized-Volatility-Forecasting) | main | 1,757 | Python | Apache-2.0 | 1 | 2026-07-21T11:15:52Z |
| [weibaozhong/BMambaXer](https://github.com/weibaozhong/BMambaXer) | main | 38 | Python | null | 2 | 2025-05-14T11:05:33Z |
| [StriveOrange/TGSINFO-VMD-NEQQLattice](https://github.com/StriveOrange/TGSINFO-VMD-NEQQLattice) | main | 12,110 | Jupyter Notebook | null | 0 | 2024-12-24T15:29:47Z |
| [maybe-collab/carbon-price-forecasting-system](https://github.com/maybe-collab/carbon-price-forecasting-system) | main | 33 | null | null | 0 | 2026-01-08T13:07:15Z |

ayk的API许可证为Other/NOASSERTION，但已读LICENSE第16–17行明确code采用MIT；数据、论文及Yahoo原始输入另有不同条款。API language为null不等于没源码：HAR-RV是Python notebook，maybe仓库ZIP内是MATLAB。VolaTTM旧别名当前指向表中的canonical full_name。

## 源码核对与可比性判定

### 1. [bashtage/arch](https://github.com/bashtage/arch)

固定commit：`cd31bc2e524a9cc128ffb6199633cf507ab24c65`。分级：**已读源码片段**，未执行。

任务/目标：通用HARX/GARCH波动率库；HARX目标由调用方给出；GARCH为收益条件方差，并非自带range预测任务。

数据：有打包示例数据，不等于完整RV benchmark输入。

依赖缺口：requirements/pyproject存在；Python>=3.10，numpy/pandas/scipy/statsmodels等。wheel/ABI或源码编译兼容未验证。

因果核对：已读片段由调用方控制训练样本；未见固定全样本分解/test早停，不能据此认证所有示例。HARX线性均值预测不保证非负，需接既定正预测约束。

移植理由：最少的数据采集/作者driver负担，容易用共同训练bank和origin控制边界；若需源码构建则工作量升M。

QLIKE/MSE：条件可；相同raw variance目标、训练长度、origin和统一评估器下重算；原生收益方差分数不可直接混比。

源码定位：[arch/univariate/mean.py](https://github.com/bashtage/arch/blob/cd31bc2e524a9cc128ffb6199633cf507ab24c65/arch/univariate/mean.py) 行477,785,786,787。

API证据：[metadata](https://api.github.com/repos/bashtage/arch) / [default branch](https://api.github.com/repos/bashtage/arch/branches/main) / [pinned recursive tree](https://api.github.com/repos/bashtage/arch/git/trees/cd31bc2e524a9cc128ffb6199633cf507ab24c65?recursive=1)，均HTTP200。关键contents也均200，且blob SHA与pinned tree相符。

### 2. [husaam-atq/volatility-risk-forecasting-platform](https://github.com/husaam-atq/volatility-risk-forecasting-platform)

固定commit：`ae7dd4781b6e039ad70b9e57d448f232ca0af3d6`。分级：**已读源码片段**，未执行。

任务/目标：HAR/稳健回归、EWMA/GARCH及range基准；默认下一日21期close-to-close滚动年化vol；含Parkinson特征，非未来h期raw range方差均值。

数据：有data/sample_prices.csv和database/market_risk.duckdb；仅树核实，未下载。

依赖缺口：requirements有numpy/pandas/scikit-learn/duckdb/arch/streamlit等；抽取回归核可去UI/数据库依赖，未查本机安装。

因果核对：HAR的Scaler在训练Pipeline内fit；walk_forward按target_date划分period。默认含最长252期特征，不能在较短等长协议中照用。原QLIKE也floor实际方差，须换统一评分。

移植理由：训练内缩放逻辑能分离，已有range特征；主要工作是替换目标、限制历史长度及评分器。

QLIKE/MSE：条件可；改为本项目raw方差及1/5/22均值目标。原作者MSE在vol单位，需重新计算，不能换算其汇总表。

源码定位：[src/volatility_platform/models/har_models.py](https://github.com/husaam-atq/volatility-risk-forecasting-platform/blob/ae7dd4781b6e039ad70b9e57d448f232ca0af3d6/src/volatility_platform/models/har_models.py) 行54,55,56；[src/volatility_platform/models/walk_forward.py](https://github.com/husaam-atq/volatility-risk-forecasting-platform/blob/ae7dd4781b6e039ad70b9e57d448f232ca0af3d6/src/volatility_platform/models/walk_forward.py) 行24,30,35；[src/volatility_platform/features/lagged_features.py](https://github.com/husaam-atq/volatility-risk-forecasting-platform/blob/ae7dd4781b6e039ad70b9e57d448f232ca0af3d6/src/volatility_platform/features/lagged_features.py) 行40；[src/volatility_platform/backtesting/forecast_metrics.py](https://github.com/husaam-atq/volatility-risk-forecasting-platform/blob/ae7dd4781b6e039ad70b9e57d448f232ca0af3d6/src/volatility_platform/backtesting/forecast_metrics.py) 行22,23,24。

API证据：[metadata](https://api.github.com/repos/husaam-atq/volatility-risk-forecasting-platform) / [default branch](https://api.github.com/repos/husaam-atq/volatility-risk-forecasting-platform/branches/main) / [pinned recursive tree](https://api.github.com/repos/husaam-atq/volatility-risk-forecasting-platform/git/trees/ae7dd4781b6e039ad70b9e57d448f232ca0af3d6?recursive=1)，均HTTP200。关键contents也均200，且blob SHA与pinned tree相符。

### 3. [ayk5511/volatility-forecasting](https://github.com/ayk5511/volatility-forecasting)

固定commit：`b86a48dec95c405970c24d0c9e7a43795b71e062`。分级：**已读源码片段**，未执行。

任务/目标：HAR/GARCH/LSTM/GRU及LightGBM/XGBoost horse race；日收益平方未来h期均值开方并年化，h=1/5/22；不是高频RV或raw Parkinson方差。

数据：有combined.parquet、spx_daily.parquet、vix_daily.parquet及输出；未读数值。

依赖缺口：未见requirements/环境锁；源码需arch/sklearn/lightgbm/xgboost/pyarrow，部分需torch，采集需yfinance；signal.alarm需Windows适配。

因果核对：03 runner以测试批次末日数据拟合GARCH再取批内conditional_volatility；HAR/tree训练切片含批次首日，其前向目标尚未成熟。02 HAR有类似边界。原runner不通过静态因果审查。

移植理由：源码、入口及数据真实存在，code许可证明确MIT；树模型提供不同于HAR的族。排名M只适用于隔离估计器。

QLIKE/MSE：估计器条件可，原runner不可。用共同driver按h隔离未成熟标签，重算variance QLIKE/MSE；不借作者结果作成绩。

源码定位：[code/03_run_core_models.py](https://github.com/ayk5511/volatility-forecasting/blob/b86a48dec95c405970c24d0c9e7a43795b71e062/code/03_run_core_models.py) 行53,64,66,92,93,117,144,165,166；[code/01_collect_data.py](https://github.com/ayk5511/volatility-forecasting/blob/b86a48dec95c405970c24d0c9e7a43795b71e062/code/01_collect_data.py) 行78,79,80,81,82,83,84。

API证据：[metadata](https://api.github.com/repos/ayk5511/volatility-forecasting) / [default branch](https://api.github.com/repos/ayk5511/volatility-forecasting/branches/main) / [pinned recursive tree](https://api.github.com/repos/ayk5511/volatility-forecasting/git/trees/b86a48dec95c405970c24d0c9e7a43795b71e062?recursive=1)，均HTTP200。关键contents也均200，且blob SHA与pinned tree相符。

### 4. [deep-hedger-Peng/HAR-RV](https://github.com/deep-hedger-Peng/HAR-RV)

固定commit：`c5da50180476838bcef059f759432b7632fa7065`。分级：**已读源码片段**，未执行。

任务/目标：单notebook HAR/OLS；分钟价格差平方聚合后开方的日vol及其下一日值，不同于本项目range proxy。

数据：不含SPY1min_clean.csv；源码另调Yahoo的SPY/VIX。

依赖缺口：无requirements；numpy/pandas/statsmodels/scipy/matplotlib/pandas_datareader及notebook magic；外部接口未核。

因果核对：创建未来Target后，对整个rv表做min-max，再60/40切分；明确使用测试期及Target全样本统计。不能保留该预处理。

移植理由：核短，但缺数据/版本且归一化泄漏；与前两项HAR重复，优先级较低。

QLIKE/MSE：条件可；重建合法训练缩放、raw variance目标及正预测协议。原生只有MSE等，无已认证QLIKE。

源码定位：[HAR-RV_forecast.ipynb](https://github.com/deep-hedger-Peng/HAR-RV/blob/c5da50180476838bcef059f759432b7632fa7065/HAR-RV_forecast.ipynb) 行19,50,82,86,92,97（notebook代码cell串联行号，非JSON物理行号）。

API证据：[metadata](https://api.github.com/repos/deep-hedger-Peng/HAR-RV) / [default branch](https://api.github.com/repos/deep-hedger-Peng/HAR-RV/branches/master) / [pinned recursive tree](https://api.github.com/repos/deep-hedger-Peng/HAR-RV/git/trees/c5da50180476838bcef059f759432b7632fa7065?recursive=1)，均HTTP200。关键contents也均200，且blob SHA与pinned tree相符。

### 5. [Aurelien7877/VolaTTM-Cross-Asset-Realized-Volatility-Forecasting](https://github.com/Aurelien7877/VolaTTM-Cross-Asset-Realized-Volatility-Forecasting)

固定commit：`cb0350080fc6c26b816cd27e3b0252a23cf74df4`。分级：**已读源码片段**，未执行。

任务/目标：TTM微调的5分钟RV预测及rolling HAR基准；log 5分钟RV路径；评估端点t+h的年化vol，不是未来h期raw variance均值。

数据：VOLARE ZIP不随仓库；loader在data/raw/volare找不到即抛错。

依赖缺口：requirements-ttm有granite-tsfm==0.3.6、torch==2.10.0、numpy>=2、pyarrow、arch；还需IBM预训练权重，均未下载。

因果核对：make_windows约束完整target日期；rolling HAR用origin-horizon截断，selection用validation，未见test早停。默认TTM context512、HAR估计窗1000均非共同等长历史；预训练时点未认证。评估实际值floor，QLIKE用vol比值。

移植理由：训练代码完整，但权重/512输入及端点vol目标与本协议差距较大；不优先直接移植主网络。

QLIKE/MSE：当前不可直接比较；HAR可统一均值目标/历史长度及variance评估后比较，TTM还需认证预训练信息边界。

源码定位：[scripts/train_volattm.py](https://github.com/Aurelien7877/VolaTTM-Cross-Asset-Realized-Volatility-Forecasting/blob/cb0350080fc6c26b816cd27e3b0252a23cf74df4/scripts/train_volattm.py) 行104,110,126,131,197,199,201,311；[src/volatility_fm/volare_data.py](https://github.com/Aurelien7877/VolaTTM-Cross-Asset-Realized-Volatility-Forecasting/blob/cb0350080fc6c26b816cd27e3b0252a23cf74df4/src/volatility_fm/volare_data.py) 行54,113。

API证据：[metadata](https://api.github.com/repos/Aurelien7877/volattm) / [default branch](https://api.github.com/repos/Aurelien7877/volattm/branches/main) / [pinned recursive tree](https://api.github.com/repos/Aurelien7877/volattm/git/trees/cb0350080fc6c26b816cd27e3b0252a23cf74df4?recursive=1)，均HTTP200。关键contents也均200，且blob SHA与pinned tree相符。

### 6. [weibaozhong/BMambaXer](https://github.com/weibaozhong/BMambaXer)

固定commit：`3548fb833d74ff0c8a6bfd17ad81cfb2419695a7`。分级：**已读源码片段**，未执行。

任务/目标：碳价多步点预测，Mamba+Transformer；点价，不是RV/range variance。

数据：无价格数据；Hubei脚本引用外部Carbon CSV。

依赖缺口：有requirements但漏掉模型import的mamba_ssm；锁torch==2.0.0等，Mamba扩展/平台/CUDA构建兼容未核。

因果核对：custom路径Scaler只fit train_data，不能说全样本标准化。每epoch计算test_loss，但EarlyStopping实际传vali_loss，不能说test早停。严格协议应移除训练时test访问；过去窗口内双向结构本身不构成未来泄漏。

移植理由：无数据、漏声明依赖和目标/正输出转换，使首轮成本高于通用RV模型。

QLIKE/MSE：原点价MSE不可比；需以共同proxy目标改head与因果driver后才可能提供统一QLIKE/MSE，运行性未验证。

源码定位：[exp/exp_long_term_forecasting.py](https://github.com/weibaozhong/BMambaXer/blob/3548fb833d74ff0c8a6bfd17ad81cfb2419695a7/exp/exp_long_term_forecasting.py) 行161,162,166；[data_provider/data_loader.py](https://github.com/weibaozhong/BMambaXer/blob/3548fb833d74ff0c8a6bfd17ad81cfb2419695a7/data_provider/data_loader.py) 行249,250,265；[models/BMambaXer.py](https://github.com/weibaozhong/BMambaXer/blob/3548fb833d74ff0c8a6bfd17ad81cfb2419695a7/models/BMambaXer.py) 行8。

API证据：[metadata](https://api.github.com/repos/weibaozhong/BMambaXer) / [default branch](https://api.github.com/repos/weibaozhong/BMambaXer/branches/main) / [pinned recursive tree](https://api.github.com/repos/weibaozhong/BMambaXer/git/trees/3548fb833d74ff0c8a6bfd17ad81cfb2419695a7?recursive=1)，均HTTP200。关键contents也均200，且blob SHA与pinned tree相符。

### 7. [StriveOrange/TGSINFO-VMD-NEQQLattice](https://github.com/StriveOrange/TGSINFO-VMD-NEQQLattice)

固定commit：`3e8a155f034f38cb20430e601c4edb13c22ed7fc`。分级：**已读源码片段**，未执行。

任务/目标：碳价分解、点/区间预测；预测notebook仅树核实；点价/价格区间来自仓库description与文件树；主预测源码未读。

数据：有Data_GuangZhou.xlsx、分解CSV及预测输出，仅树核实。

依赖缺口：MATLAB及仓库内VMD/TGSINFO函数；未见依赖清单。大notebook未取，不能据旧记录宣称feyn已重新确认。

因果核对：脚本翻转数据后取前1522点整体优化/VMD，未见逐origin截断。若train/val/test落在该块内则跨边界；大notebook未读，具体对应未认证。旧“全序列”只能限定固定块，不能说整个工作簿均已证明泄漏。

移植理由：全块预处理需重构，主预测notebook8.9MB未读，不能认证完整模型可跑。

QLIKE/MSE：暂不能承诺。需先审预测源码/依赖，再实现每origin只用等长过去的分解及共同variance目标；区间输出不等同方差点预测。

NEQQLattice预测notebook的等级只有**仅静态核实**。本轮不读取8,887,431bytes的大notebook，不能用已读MATLAB分解入口替代预测源码审查，也不能把feyn依赖写成已重新确认。

源码定位：[TGSINFO-VMD/PE_Cost.m](https://github.com/StriveOrange/TGSINFO-VMD-NEQQLattice/blob/3e8a155f034f38cb20430e601c4edb13c22ed7fc/TGSINFO-VMD/PE_Cost.m) 行6；[TGSINFO-VMD/TGSINFO_VMD_GZ.m](https://github.com/StriveOrange/TGSINFO-VMD-NEQQLattice/blob/3e8a155f034f38cb20430e601c4edb13c22ed7fc/TGSINFO-VMD/TGSINFO_VMD_GZ.m) 行6,7,10,11,12,25,39。

API证据：[metadata](https://api.github.com/repos/StriveOrange/TGSINFO-VMD-NEQQLattice) / [default branch](https://api.github.com/repos/StriveOrange/TGSINFO-VMD-NEQQLattice/branches/main) / [pinned recursive tree](https://api.github.com/repos/StriveOrange/TGSINFO-VMD-NEQQLattice/git/trees/3e8a155f034f38cb20430e601c4edb13c22ed7fc?recursive=1)，均HTTP200。关键contents也均200，且blob SHA与pinned tree相符。

### 8. [maybe-collab/carbon-price-forecasting-system](https://github.com/maybe-collab/carbon-price-forecasting-system)

固定commit：`2971f905625da81fd8311c1dfedefd13d5bba5b6`。分级：**已读源码片段**，未执行。

任务/目标：ZIP内MATLAB MVMD+多模型碳价点预测；多变量价格点值，不是RV/range variance。

数据：顶层仅41,725bytes ZIP；内含MATLAB入口，但当前ZIP工作文件无GDEA.xls。

依赖缺口：MATLAB及BPNN/CNN/SVR所需toolbox/接口需核；无环境清单。

因果核对：mian.m行8读GDEA.xls；行25先对完整x做MVMD，行80以后才形成训练/测试切分。直接以这批分解特征评估未来会引入未来信息；缺工作簿也阻止原样运行。

移植理由：静态确认有源码，旧记录应改为“有MATLAB入口、缺数据”；整体分解和环境仍需重构。

QLIKE/MSE：原价格分数不可比；须补数据适配、改因果分解及共同variance目标/评分，不是即用复现。

ZIP只有41,725bytes，已在内存查看目录与三个.m源文件；没有解压到磁盘或读取.git配置。实际入口名为mian.m。其第8行引用GDEA.xls，但当前ZIP工作文件中没有该表。旧记录须区分“没有代码”和“有入口但缺数据”。

源码定位：[carbon price forecasting system.zip](https://api.github.com/repos/maybe-collab/carbon-price-forecasting-system/contents/carbon%20price%20forecasting%20system.zip?ref=2971f905625da81fd8311c1dfedefd13d5bba5b6) 内mian.m行8/25/80–83。

API证据：[metadata](https://api.github.com/repos/maybe-collab/carbon-price-forecasting-system) / [default branch](https://api.github.com/repos/maybe-collab/carbon-price-forecasting-system/branches/main) / [pinned recursive tree](https://api.github.com/repos/maybe-collab/carbon-price-forecasting-system/git/trees/2971f905625da81fd8311c1dfedefd13d5bba5b6?recursive=1)，均HTTP200。关键contents也均200，且blob SHA与pinned tree相符。

## 首轮优先的三项

1. **arch HARX接口。**先建立最小可控的库级基准，样本切片、标签成熟和预测origin交由我们的driver掌握。若使用GARCH，须另列其收益方差信息集；包运行兼容仍待实测。
2. **husaam的log-HAR Ridge/Huber核。**训练内缩放与回归可分离，且已有Parkinson特征。替换默认21期vol目标、252期历史特征和评分器，不移植数据库或仪表盘。
3. **ayk的LightGBM/XGBoost估计器。**增加不同于HAR的模型族，源码与入口实际存在、code明确MIT。只用估计器接我们的因果driver，不沿用其测试批次拟合与未成熟标签训练流程。

## 共同协议与边界

- 使用已冻结的共同origin、等长历史L、训练bank和标签成熟规则；不能一项用512期、另一项用1000期或任意expanding历史。此轮不新挑lookback、超参或模型。
- 原生点价、年化vol、端点RV与未来h期raw range variance均值不互换。先统一目标/单位，再用本项目冻结评估器重算QLIKE/MSE；作者汇总表不是本项目成绩。
- 保留raw零目标。不能删零或抬高零来凑QLIKE；某些作者指标floor实际值，必须换掉。如既定协议采用Q0，log(pred)+y/pred在y=0有效，本轮不另定指标。
- 普通训练batch shuffle、WeightedRandomSampler和过去窗口内双向编码不自动构成未来泄漏。审查的是时间切分、缩放fit范围、标签成熟、分解截断与选择是否读取未来。
- 依赖缺失/未核指未附、未声明或需外部资源，未检查本机是否已装。没有pip install、作者脚本执行、训练、模型权重或数据文件下载。
- 大于5MB的arch、husaam和TGS仅读API元数据与小型keyfiles；最大单个keyfile282,219bytes。未读取8.9MB notebook、数据库、数据表或权重。
- 未读取本项目2026数据/指标，未改旧文件；只新增MD/JSON。公开可见、已读源码与运行通过是不同状态，本轮只到前两项。
