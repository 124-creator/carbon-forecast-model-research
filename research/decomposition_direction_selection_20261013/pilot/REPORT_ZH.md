# 有界因果分解实际试验

不是VMD/CEEMDAN，也不是新分解算法。仅广州作者报告价格；无外生变量。2020—2021已用开发数据，451同年完整目标起点，1/5/20观察点预测。所有结果为原价平方误差的统一相对RW评分。

| 配置 | 评分 |
|---|---|
| RW | 1.00000000 |
| MLE480 | 0.94979323 |
| RAW8_a0.1 | 1.02961724 |
| RAW16_a0.1 | 1.04241859 |
| DECOMP_w5_a0.1 | 1.06227327 |
| MATURE_w5_a0.1 | 1.06523172 |
| DECOMP_w10_a0.1 | 1.06421343 |
| MATURE_w10_a0.1 | 1.12133127 |
| DECOMP_w20_a0.1 | 1.06911279 |
| MATURE_w20_a0.1 | 1.12353474 |
| RAW8_a1.0 | 1.01479773 |
| RAW16_a1.0 | 1.01461535 |
| DECOMP_w5_a1.0 | 1.04697070 |
| MATURE_w5_a1.0 | 1.03961905 |
| DECOMP_w10_a1.0 | 1.03386076 |
| MATURE_w10_a1.0 | 1.05099976 |
| DECOMP_w20_a1.0 | 1.03490819 |
| MATURE_w20_a1.0 | 1.04066059 |
| RAW8_a10.0 | 1.04326691 |
| RAW16_a10.0 | 1.04293227 |
| DECOMP_w5_a10.0 | 1.04848376 |
| MATURE_w5_a10.0 | 1.04693978 |
| DECOMP_w10_a10.0 | 1.04769569 |
| MATURE_w10_a10.0 | 1.05123884 |
| DECOMP_w20_a10.0 | 1.04860545 |
| MATURE_w20_a10.0 | 1.04941293 |

## 机制判断
[
  {
    "w": 5,
    "alpha": 1.0,
    "by_year": [
      {
        "year": "2020",
        "count": 224,
        "future_revision_diagnostic_count": 224,
        "revision_rms": 0.19988079797450348,
        "increment_rms": 0.430457086794292,
        "revision_ratio": 0.46434546928489306,
        "association_h": [
          {
            "pearson": 0.09914048062051085,
            "spearman": -0.0796519325218877
          },
          {
            "pearson": 0.28443762657040134,
            "spearman": 0.19938073884262225
          },
          {
            "pearson": 0.5001386281702841,
            "spearman": 0.4167435404655136
          }
        ]
      },
      {
        "year": "2021",
        "count": 227,
        "future_revision_diagnostic_count": 227,
        "revision_rms": 0.49763302895740963,
        "increment_rms": 0.6480808673095414,
        "revision_ratio": 0.7678563803669987,
        "association_h": [
          {
            "pearson": -0.0285892913907656,
            "spearman": -0.045304481819730034
          },
          {
            "pearson": -0.08112760264393028,
            "spearman": -0.10387201483885927
          },
          {
            "pearson": -0.2783234314060431,
            "spearman": -0.19693497124355472
          }
        ]
      }
    ],
    "bootstrap_mature_minus_mle": [
      -0.06515439064422075,
      0.0979962871901486,
      0.33185561081573667
    ],
    "bootstrap_mature_minus_decomp": [
      -0.05178171809836661,
      -0.007764954686736629,
      0.018861384961787655
    ],
    "H1": true,
    "H2": false,
    "H3": false,
    "mechanism_established": false
  },
  {
    "w": 10,
    "alpha": 1.0,
    "by_year": [
      {
        "year": "2020",
        "count": 224,
        "future_revision_diagnostic_count": 224,
        "revision_rms": 0.25892860126761635,
        "increment_rms": 0.430457086794292,
        "revision_ratio": 0.6015201264216934,
        "association_h": [
          {
            "pearson": 0.14972132592249776,
            "spearman": 0.06275891522528296
          },
          {
            "pearson": 0.35454221969412303,
            "spearman": 0.26106982703395265
          },
          {
            "pearson": 0.602740248200169,
            "spearman": 0.5732062780269058
          }
        ]
      },
      {
        "year": "2021",
        "count": 227,
        "future_revision_diagnostic_count": 227,
        "revision_rms": 0.8215567191519153,
        "increment_rms": 0.6480808673095414,
        "revision_ratio": 1.2676762431863569,
        "association_h": [
          {
            "pearson": -0.024914575959990966,
            "spearman": -0.04293666605795609
          },
          {
            "pearson": -0.0852655732433525,
            "spearman": -0.129228572190681
          },
          {
            "pearson": -0.15279348080750813,
            "spearman": 0.019223627272149033
          }
        ]
      }
    ],
    "bootstrap_mature_minus_mle": [
      -0.037010924946458225,
      0.11081459975718938,
      0.3711217825725394
    ],
    "bootstrap_mature_minus_decomp": [
      -0.002140690467078215,
      0.017041077655646657,
      0.04089757852851274
    ],
    "H1": true,
    "H2": false,
    "H3": false,
    "mechanism_established": false
  },
  {
    "w": 20,
    "alpha": 1.0,
    "by_year": [
      {
        "year": "2020",
        "count": 224,
        "future_revision_diagnostic_count": 224,
        "revision_rms": 0.3445726021983076,
        "increment_rms": 0.430457086794292,
        "revision_ratio": 0.8004807279731763,
        "association_h": [
          {
            "pearson": 0.144989964261018,
            "spearman": 0.13673606662395904
          },
          {
            "pearson": 0.2557979000011783,
            "spearman": 0.2933247918001282
          },
          {
            "pearson": 0.5865546457753981,
            "spearman": 0.6459160794362588
          }
        ]
      },
      {
        "year": "2021",
        "count": 227,
        "future_revision_diagnostic_count": 227,
        "revision_rms": 1.4282886874991376,
        "increment_rms": 0.6480808673095414,
        "revision_ratio": 2.2038741761172025,
        "association_h": [
          {
            "pearson": -0.05454318514841709,
            "spearman": -0.06039776842597704
          },
          {
            "pearson": -0.016280834082142964,
            "spearman": -0.04822526668704822
          },
          {
            "pearson": 0.04150724743393335,
            "spearman": 0.029607956189252913
          }
        ]
      }
    ],
    "bootstrap_mature_minus_mle": [
      -0.03193269706570561,
      0.09781412410701254,
      0.2944697327134123
    ],
    "bootstrap_mature_minus_decomp": [
      -0.012350055198457834,
      0.0075474589546869345,
      0.036177397157530336
    ],
    "H1": true,
    "H2": false,
    "H3": false,
    "mechanism_established": false
  }
]

未建立机制时只说明当前数据条件不足，不升级注意力。分解8+残差8和RAW16同维同Ridge，MATURE只追加2项历史已成熟修订信息；RAW8是重构信息相同但维数不同的补充基线。训练只使用原价增量目标，未替换为分量损失。future_revision是事后诊断，不作当前特征。有限样本关联不代表因果关系；alpha及窗口搜索均为开发探索，置信区间未校正多重搜索。MLE采用更多近期单步成熟标签，信息支持不同。bootstrap为年份内循环20点块，500次，种子20261013。预测数组只供本地审核。