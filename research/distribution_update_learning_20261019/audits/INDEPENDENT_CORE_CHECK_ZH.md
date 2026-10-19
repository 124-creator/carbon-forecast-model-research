# 独立核心数值与数学审计

日期：2026-10-19。仅合成数值；不读市场，不拟合预测模型。

总体状态：通过。默认核验cap_fraction=0.25，并比较cap_fraction=1消融。

## 检查方法

GMM CRPS直接积分CDF平方定义；a与b分别直接积分，不用源代码的自距离恒等式作唯一参考。混合alpha对照组合CDF数值积分。原子CRPS对照完整双重绝对距离和。

对数正态CRPS在正实数支持上用log坐标积分CDF定义，报告积分误差及省略尾部的显式上界；另外检查正价格/正尺度和有限均值。

反射控制用数值Gaussian continuation积分与独立标量实现核验；枚举32条幅度依赖真实过去符号的路径，逐步隔离clip后的奇性，并检验总幅度、强符号偏差预算、h1与no-prefix零控制。

box-QP对照scipy SLSQP，包含PSD奇异矩阵、不同尺度及1/2/3维；只比较应唯一的目标值，不强求非唯一最优权重相同。

bootstrap用独立重采样频率矩阵重建统计量；偏斜单异常值fixture使B或K的符号反转产生不同临界值。矩阵上界检查全部64个对称误差盒顶点。

## 结果

```json
{
  "scores": {
    "status": "passed",
    "normal_mixture_fixtures": 6,
    "mixed_alpha_checks": 30,
    "CRPS_CDF_integral_max_abs_error": 6.661338147750939e-16,
    "a_b_direct_CDF_integral_max_abs_error": 4.579669976578771e-16,
    "mixture_CRPS_quadratic_max_abs_error": 6.661338147750939e-16,
    "W1_CDF_integral_max_abs_error": 2.220446049250313e-16,
    "batched_CRPS_max_abs_error": 1.1102230246251565e-16,
    "atomic_direct_pair_sum_max_abs_error": 5.551115123125783e-16,
    "quantile_CDF_max_abs_error": 1.1102230246251565e-16,
    "normal_integrator_max_reported_error": 3.835813487932355e-11,
    "lognormal_fixtures": 4,
    "lognormal_CDF_integral_max_abs_error": 1.5631940186722204e-13,
    "lognormal_batched_max_abs_error": 2.842170943040401e-14,
    "lognormal_max_omitted_tail_bound": 7.425004031629522e-26,
    "lognormal_integrator_max_reported_error": 5.024140301228561e-10,
    "lognormal_invalid_nonpositive_inputs_rejected": 3,
    "domain": "Positive finite scales, normalized nonnegative weights; centered normal mixtures, finite-mean lognormal mixtures. No arbitrary mean normal-mixture API is claimed."
  },
  "controls": {
    "status": "passed",
    "default_checked_cap_fraction": 0.25,
    "Gaussian_continuation_quadrature_max_abs_error": 1.3530843112619095e-15,
    "clipped_increment_oddness_max_abs_error": 2.7755575615628914e-17,
    "h1_zero_center_scale_control_max_abs": 3.3306690738754695e-17,
    "no_prefix_control_max_abs": 5.551115123125783e-17,
    "batched_vs_single_origin_max_abs_error": 0.0,
    "leveraged_path_enumerations": [
      {
        "cap_fraction": 0.25,
        "sign_imbalance": 0.0,
        "E_control": -2.6020852139652106e-18,
        "delta_E_spent": 0.0,
        "delta_cap": 0.0,
        "clipped_step_count_reference": 22,
        "path_count": 32
      },
      {
        "cap_fraction": 0.25,
        "sign_imbalance": 0.05,
        "E_control": 0.0003755064065707322,
        "delta_E_spent": 0.003443284004499854,
        "delta_cap": 0.003949528575974183,
        "clipped_step_count_reference": 22,
        "path_count": 32
      },
      {
        "cap_fraction": 0.25,
        "sign_imbalance": 0.1,
        "E_control": 0.0012757815798031683,
        "delta_E_spent": 0.006885957397113902,
        "delta_cap": 0.007899057151948366,
        "clipped_step_count_reference": 22,
        "path_count": 32
      },
      {
        "cap_fraction": 1.0,
        "sign_imbalance": 0.0,
        "E_control": -4.336808689942018e-19,
        "delta_E_spent": 0.0,
        "delta_cap": 0.0,
        "clipped_step_count_reference": 0,
        "path_count": 32
      },
      {
        "cap_fraction": 1.0,
        "sign_imbalance": 0.05,
        "E_control": 9.6760851431493e-05,
        "delta_E_spent": 0.0039038975879944204,
        "delta_cap": 0.015798114303896733,
        "clipped_step_count_reference": 0,
        "path_count": 32
      },
      {
        "cap_fraction": 1.0,
        "sign_imbalance": 0.1,
        "E_control": 0.0008192830941516067,
        "delta_E_spent": 0.007722056141861767,
        "delta_cap": 0.031596228607793465,
        "clipped_step_count_reference": 0,
        "path_count": 32
      }
    ]
  },
  "box_quadratic": {
    "status": "passed",
    "quadratic_cases": 54,
    "dimensions": [
      1,
      2,
      3
    ],
    "scipy_reference": "SLSQP with analytic derivative and three starts; objective scaled for numeric conditioning",
    "normalized_objective_max_abs_difference": 2.5687747319882043e-16,
    "maximum_box_constraint_excess": 0.0,
    "minimum_generated_eigenvalue_roundoff": -4.352399258787115e-26
  },
  "block_bootstrap": {
    "status": "passed",
    "manual_frequency_bootstrap_max_abs_difference": 2.6645352591003757e-15,
    "joint_raw_baseline_uses_identical_quantile": true,
    "Gershgorin_all_symmetric_corner_count": 64,
    "Gershgorin_minimum_eigenvalue_margin": 6.505213034913027e-19,
    "asymmetric_direction_fixtures": [
      {
        "fixture": "B_lower",
        "correct_quantile": 2.059136524968903,
        "incorrect_reversed_quantile": 1.0295682624844515
      },
      {
        "fixture": "K_upper",
        "correct_quantile": 1.0239273039876275,
        "incorrect_reversed_quantile": 2.047854607975255
      }
    ],
    "scope": "Correct signs and deterministic construction checked. Circular-block bootstrap remains approximate; no iid/distribution-free coverage theorem is inferred."
  },
  "empirical_bernstein": {
    "status": "passed",
    "independent_formula_max_abs_difference": 0.0,
    "constant_zero_range_bound": 0.0
  }
}
```

## 数学审查

审计开发记录：第一轮其余四组通过，控制组使用浮点精确等号检查理论零值，在约1e−16的normal-CDF舍入残差处失败。定位后仅将审计该断言改为1e−14绝对容差，源代码未改，其余容差未放宽。重跑为审计修正，不计新的独立市场实验。

B下界的bootstrap方向为bootstrap mean−empirical mean；K上界方向相反。A使用双侧误差半宽e，逐元素误差盒通过行和对角项得到Loewner上界。源码方向正确；该结论不把block bootstrap升级为有限样本分布自由保证。

predictable remaining cap加对称clip保持当前增量奇性。δ×cap的偏差界依赖给定当前幅度与过去的条件符号偏差假设；普通反射TV不够。K(X)变化时总体上界涉及E K(X)，源码block_gate对cap×basis另建upper_K，这是正确处理方向。

F0/F1同中心对称时，osc(B)≤W1/2，单步reflection绝对值≤W1/4；把整个路径总预算设为W1/4是合法而更紧的设计，但会截断多个贡献，不是无需实验的最优性定理。W1整额消融应保留。

参考方差代码为(h−j−1)/h，表示每步方差1/h、终点总参考方差1。若注释仍写unit-variance Gaussian increments，应改成上述精确表述；这不影响奇性。

适用域：正有限尺度、非负归一混合权重。当前GMM API是零中心而非任意混合均值。积分/优化核验只覆盖所列数值域，不能代替后续pipeline对非有限值和输入约束的硬检查。

## 文件身份

```json
{
  "date": "2026-10-19",
  "seed": 20261019,
  "cap_fraction": 0.25,
  "script_sha256": "d40f9a4095a92b23dd49510f928e7aab0d4baa1eb334eb6a85b96d1c63201c9e",
  "source_hashes": {
    "scores.py": "a6c12bc76b03966db8b61f99e1c122c2a49df60195fb24662b9828733dc8b5e8",
    "learning.py": "b268c3b22900a8a5a1844c98f55d1433baa0941245e0fa977dbdf1ca8468a996"
  },
  "source_hashes_after": {
    "scores.py": "a6c12bc76b03966db8b61f99e1c122c2a49df60195fb24662b9828733dc8b5e8",
    "learning.py": "b268c3b22900a8a5a1844c98f55d1433baa0941245e0fa977dbdf1ca8468a996"
  },
  "numpy_version": "2.3.5",
  "market_rows_read": 0,
  "predictive_model_fits": 0
}
```

集中界来源：Maurer & Pontil (2009), Theorem4 / Corollary5, https://arxiv.org/pdf/0907.3740 。其他数学恒等式由本次外部积分与代数审计交叉核验。
