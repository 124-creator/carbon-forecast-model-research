# Forest prior round5 cache接口

每市场一个文件，文件名前缀为 `forest_prior_round5_`，市场为 CEA/GUANGZHOU/SHANGHAI/HUBEI/EU，后缀 `.pt`。全部tensor存CPU，顶层仅dict/list/内置标量与tensor。

```python
from forest_prior_round5 import load_cache
phase = load_cache(market, 'inner', D)  # 或 final
bank = phase['banks'][hi]
u = bank['u'][X['sid'][rows]]           # [query_batch, NZ_bank]
```

`load_cache`核对D的t/hi/y/scale/tr/va/rf/te及全局sid。它只读cache，不训练或写任何文件。

- 顶层 `times`/`unique_times`/`origin_times`：`np.unique(D['t'])`，严格等同seq_transport.prepare全体unique origin序。
- 顶层 `row_sid`/`sid`：D每行对应上述sid，等同`np.unique(..., return_inverse=True)`；各h和两个phase共用。
- `rows`：t/hi/y/scale/z/local/x_state/x_aug/tr/va/rf/te，保留D原顺序。
- `valid_sid_h[Nsid,4]`：该sid是否存在对应h的D row；不存在者仅存成熟uniform参考并标记不用。
- `phases.inner`：fit=D.tr，boundary=validation_start_index。scaler与forest仅fit这些标签。
- `phases.final`：fit=D.rf，boundary=cut_index。scaler与forest仅fit全部合法precut标签。
- phase含 `fit_D_indices`/`fit_indices`、`scaler={'mean','std'}`、全D行的归一化/clipped `x`、sid/times别名和 `banks[4]`。

每个bank保留T.make_bank原键：

- `z[Nbank]`、`phi[Nbank,5]`、`mature[Nbank]`、`support[Nbank+1]`、`zero_pos`、`p`、`n`。
- `bank_D_indices`/`D_indices`：非零支持对应的原D行，排序严格采用raw-y非零筛选后对D.z(float64)做stable argsort，与T.make_bank一致。重复z只按D行索引映射，不按数值映射。
- `bank_t`、`bank_horizon`、`p_static`（与`p`同值）。

新增参考数据：

- `u[Nsid,Nbank]`：平滑参考，`.95*u_forest+.05*uniform`。
- `forest`/`u_forest[Nsid,Nbank]`：五seed原始leaf概率先平均，再去零/成熟mask后条件归一的未平滑参考；两键共享同tensor存储。
- `uniform[Nsid,Nbank]`：只在该query已成熟NZ支持上的均匀参考。
- `allowed`/`mature_mask[Nsid,Nbank]`：`bank_mature <= unique_times`，别名共享存储。
- `count`/`mature_count[Nsid]`：成熟NZ数。
- `m0[Nsid]`、`v0[Nsid]`：基于平滑u的条件均值/方差，非uniform的矩。
- `phi2=phi[:,:2]`。
- `valid_sid[Nsid]`、`fallback[Nsid]`、`eligible_sid=valid_sid & (count>=32)`。

无成熟NZ的sid，所有u/forest/uniform均为0，不能用于训练；沿用seq的count>=32过滤。leaf在成熟NZ上质量为0时fallback成熟NZ uniform。没有该h目标的sid也以成熟uniform标记无效，不能当有效训练/评估row。

森林和scaler/phi使用完整允许fit partition；成熟mask只限制可检索响应，不宣称每个训练query都使用当时冻结的全参数。

训练不应使用final forest给inner validation提供参考。两phase的bank长度、scaler和森林都可不同，必须按phase读取。

最终test未平滑u_forest按相同p_rule组合时，生成器已按原D索引重排并匹配旧QRF NZ weights；gamma0完整CRPS与冻结rule-QRF比较写入baseline_checks CSV。CDF边界的量化分位差异单独记录，不用其挑模型。
