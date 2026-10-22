# 市场×期限完整对照

直接读取正式JSON；候选固定为rolling_neural_prior_crps，没有按test改候选或选择seed。static/rule均为同结构CRPS损失对照。经验基准MQS为线性插值分位数，完整CRPS来自全样本ECDF，Brier为Beta平滑概率，三种经验表示需分别解释。

| market | h | model | n | MQS | coverage80 | coverage95 | Brier |
|---|---:|---|---:|---:|---:|---:|---:|
| CEA | 1 | rolling_neural_prior_crps | 332 | 0.007778319 | 0.885542 | 0.954819 | 0.084829590 |
| CEA | 1 | static_prior_crps | 332 | 0.007446248 | 0.855422 | 0.942771 | 0.151856221 |
| CEA | 1 | rolling_rule_prior_crps | 332 | 0.007785052 | 0.885542 | 0.954819 | 0.084619412 |
| CEA | 1 | empirical_constant | 332 | 0.007325422 | 0.777108 | 0.930723 | 0.151856204 |
| CEA | 5 | rolling_neural_prior_crps | 330 | 0.017731425 | 0.760606 | 0.863636 | 0.007457944 |
| CEA | 5 | static_prior_crps | 330 | 0.017710402 | 0.745455 | 0.860606 | 0.034636642 |
| CEA | 5 | rolling_rule_prior_crps | 330 | 0.017732737 | 0.760606 | 0.866667 | 0.007385365 |
| CEA | 5 | empirical_constant | 330 | 0.018004517 | 0.633333 | 0.845455 | 0.034636636 |
| CEA | 10 | rolling_neural_prior_crps | 327 | 0.027571949 | 0.651376 | 0.795107 | 0.016057888 |
| CEA | 10 | static_prior_crps | 327 | 0.027748581 | 0.620795 | 0.792049 | 0.034623662 |
| CEA | 10 | rolling_rule_prior_crps | 327 | 0.027571374 | 0.651376 | 0.795107 | 0.016001241 |
| CEA | 10 | empirical_constant | 327 | 0.028104623 | 0.541284 | 0.816514 | 0.034623664 |
| CEA | 20 | rolling_neural_prior_crps | 322 | 0.046775379 | 0.555901 | 0.739130 | 0.003999662 |
| CEA | 20 | static_prior_crps | 322 | 0.047156454 | 0.524845 | 0.726708 | 0.016514190 |
| CEA | 20 | rolling_rule_prior_crps | 322 | 0.046773215 | 0.555901 | 0.739130 | 0.003958993 |
| CEA | 20 | empirical_constant | 322 | 0.047766160 | 0.447205 | 0.773292 | 0.016514189 |
| GUANGZHOU | 1 | rolling_neural_prior_crps | 330 | 0.016520358 | 0.790909 | 0.881818 | 0.150604484 |
| GUANGZHOU | 1 | static_prior_crps | 330 | 0.016884609 | 0.800000 | 0.881818 | 0.175935920 |
| GUANGZHOU | 1 | rolling_rule_prior_crps | 330 | 0.016540207 | 0.790909 | 0.881818 | 0.151285227 |
| GUANGZHOU | 1 | empirical_constant | 330 | 0.016464158 | 0.818182 | 0.918182 | 0.175935901 |
| GUANGZHOU | 5 | rolling_neural_prior_crps | 328 | 0.026188494 | 0.865854 | 0.942073 | 0.012192406 |
| GUANGZHOU | 5 | static_prior_crps | 328 | 0.026207105 | 0.865854 | 0.942073 | 0.012099940 |
| GUANGZHOU | 5 | rolling_rule_prior_crps | 328 | 0.026192301 | 0.865854 | 0.942073 | 0.012179883 |
| GUANGZHOU | 5 | empirical_constant | 328 | 0.025349275 | 0.826220 | 0.960366 | 0.012099941 |
| GUANGZHOU | 10 | rolling_neural_prior_crps | 326 | 0.030537696 | 0.907975 | 0.957055 | 0.006124322 |
| GUANGZHOU | 10 | static_prior_crps | 326 | 0.030518818 | 0.907975 | 0.957055 | 0.006097401 |
| GUANGZHOU | 10 | rolling_rule_prior_crps | 326 | 0.030542078 | 0.907975 | 0.957055 | 0.006119771 |
| GUANGZHOU | 10 | empirical_constant | 326 | 0.028758460 | 0.883436 | 0.963190 | 0.006097401 |
| GUANGZHOU | 20 | rolling_neural_prior_crps | 321 | 0.042038998 | 0.900312 | 0.940810 | 0.009290260 |
| GUANGZHOU | 20 | static_prior_crps | 321 | 0.042032513 | 0.900312 | 0.940810 | 0.009299359 |
| GUANGZHOU | 20 | rolling_rule_prior_crps | 321 | 0.042043719 | 0.900312 | 0.940810 | 0.009289006 |
| GUANGZHOU | 20 | empirical_constant | 321 | 0.039862923 | 0.878505 | 0.956386 | 0.009299359 |
| SHANGHAI | 1 | rolling_neural_prior_crps | 130 | 0.013433716 | 0.723077 | 0.838462 | 0.123298097 |
| SHANGHAI | 1 | static_prior_crps | 130 | 0.013788084 | 0.715385 | 0.823077 | 0.113782136 |
| SHANGHAI | 1 | rolling_rule_prior_crps | 130 | 0.013497711 | 0.730769 | 0.838462 | 0.115179672 |
| SHANGHAI | 1 | empirical_constant | 130 | 0.013760715 | 0.784615 | 0.861538 | 0.113782135 |
| SHANGHAI | 5 | rolling_neural_prior_crps | 128 | 0.021476822 | 0.765625 | 0.875000 | 0.065667982 |
| SHANGHAI | 5 | static_prior_crps | 128 | 0.021907516 | 0.765625 | 0.859375 | 0.066085346 |
| SHANGHAI | 5 | rolling_rule_prior_crps | 128 | 0.021508877 | 0.757812 | 0.875000 | 0.066322979 |
| SHANGHAI | 5 | empirical_constant | 128 | 0.021841576 | 0.773438 | 0.890625 | 0.066085346 |
| SHANGHAI | 10 | rolling_neural_prior_crps | 125 | 0.026083610 | 0.768000 | 0.872000 | 0.008419648 |
| SHANGHAI | 10 | static_prior_crps | 125 | 0.026921083 | 0.768000 | 0.872000 | 0.008216403 |
| SHANGHAI | 10 | rolling_rule_prior_crps | 125 | 0.026139961 | 0.776000 | 0.872000 | 0.008161865 |
| SHANGHAI | 10 | empirical_constant | 125 | 0.027898878 | 0.768000 | 0.904000 | 0.008216404 |
| SHANGHAI | 20 | rolling_neural_prior_crps | 120 | 0.035538005 | 0.758333 | 0.850000 | 0.008788984 |
| SHANGHAI | 20 | static_prior_crps | 120 | 0.036302664 | 0.758333 | 0.850000 | 0.008275428 |
| SHANGHAI | 20 | rolling_rule_prior_crps | 120 | 0.035626807 | 0.750000 | 0.850000 | 0.008526044 |
| SHANGHAI | 20 | empirical_constant | 120 | 0.034825105 | 0.758333 | 0.875000 | 0.008275428 |
| HUBEI | 1 | rolling_neural_prior_crps | 146 | 0.010726633 | 0.767123 | 0.856164 | 0.051440262 |
| HUBEI | 1 | static_prior_crps | 146 | 0.010748815 | 0.767123 | 0.856164 | 0.051910921 |
| HUBEI | 1 | rolling_rule_prior_crps | 146 | 0.010734615 | 0.767123 | 0.856164 | 0.051362034 |
| HUBEI | 1 | empirical_constant | 146 | 0.010852572 | 0.719178 | 0.958904 | 0.051910921 |
| HUBEI | 5 | rolling_neural_prior_crps | 144 | 0.014717845 | 0.840278 | 0.923611 | 0.007050834 |
| HUBEI | 5 | static_prior_crps | 144 | 0.014730746 | 0.840278 | 0.923611 | 0.006916920 |
| HUBEI | 5 | rolling_rule_prior_crps | 144 | 0.014724994 | 0.840278 | 0.923611 | 0.006965150 |
| HUBEI | 5 | empirical_constant | 144 | 0.014656528 | 0.798611 | 0.972222 | 0.006916920 |
| HUBEI | 10 | rolling_neural_prior_crps | 141 | 0.016840996 | 0.865248 | 0.957447 | 0.014157537 |
| HUBEI | 10 | static_prior_crps | 141 | 0.016832988 | 0.865248 | 0.957447 | 0.013990201 |
| HUBEI | 10 | rolling_rule_prior_crps | 141 | 0.016841032 | 0.865248 | 0.957447 | 0.014108564 |
| HUBEI | 10 | empirical_constant | 141 | 0.016705676 | 0.815603 | 1.000000 | 0.013990201 |
| HUBEI | 20 | rolling_neural_prior_crps | 136 | 0.020854593 | 0.875000 | 0.933824 | 0.007367512 |
| HUBEI | 20 | static_prior_crps | 136 | 0.020869969 | 0.875000 | 0.933824 | 0.007310913 |
| HUBEI | 20 | rolling_rule_prior_crps | 136 | 0.020863361 | 0.875000 | 0.933824 | 0.007357707 |
| HUBEI | 20 | empirical_constant | 136 | 0.020605732 | 0.889706 | 0.992647 | 0.007310912 |
| EU | 1 | rolling_neural_prior_crps | 159 | 0.010708602 | 0.786164 | 0.943396 | 0.006279447 |
| EU | 1 | static_prior_crps | 159 | 0.010701134 | 0.786164 | 0.943396 | 0.006250358 |
| EU | 1 | rolling_rule_prior_crps | 159 | 0.010708780 | 0.786164 | 0.943396 | 0.006277752 |
| EU | 1 | empirical_constant | 159 | 0.010724130 | 0.742138 | 0.962264 | 0.006250358 |
| EU | 5 | rolling_neural_prior_crps | 157 | 0.025132148 | 0.764331 | 0.929936 | 0.000109769 |
| EU | 5 | static_prior_crps | 157 | 0.025131051 | 0.764331 | 0.929936 | 0.000003144 |
| EU | 5 | rolling_rule_prior_crps | 157 | 0.025132333 | 0.764331 | 0.929936 | 0.000094479 |
| EU | 5 | empirical_constant | 157 | 0.025507970 | 0.694268 | 0.929936 | 0.000003144 |
| EU | 10 | rolling_neural_prior_crps | 155 | 0.036585052 | 0.754839 | 0.883871 | 0.000056054 |
| EU | 10 | static_prior_crps | 155 | 0.036583873 | 0.754839 | 0.890323 | 0.000003166 |
| EU | 10 | rolling_rule_prior_crps | 155 | 0.036584743 | 0.754839 | 0.883871 | 0.000048545 |
| EU | 10 | empirical_constant | 155 | 0.037442725 | 0.670968 | 0.909677 | 0.000003166 |
| EU | 20 | rolling_neural_prior_crps | 150 | 0.045561651 | 0.780000 | 0.906667 | 0.000140993 |
| EU | 20 | static_prior_crps | 150 | 0.045579389 | 0.780000 | 0.906667 | 0.000012893 |
| EU | 20 | rolling_rule_prior_crps | 150 | 0.045563611 | 0.780000 | 0.906667 | 0.000122637 |
| EU | 20 | empirical_constant | 150 | 0.046465263 | 0.733333 | 0.906667 | 0.000012893 |

## 两seed各自的全部对照方向

以下均由同一个seed自己的分布参数反CDF计算候选与网络对照；经验/Gaussian对照为无seed常数。没有根据test挑选seed。改善为全部20单元等权几何改善，正值越好。

| seed | reference | MQS改善 | 胜出/20 |
|---|---|---:|---:|
| 11 | static_prior_nll | +3.39123% | 17/20 |
| 11 | rolling_rule_prior_nll | +3.27312% | 15/20 |
| 11 | rolling_neural_prior_nll | +3.45088% | 16/20 |
| 11 | static_prior_crps | +0.70239% | 13/20 |
| 11 | rolling_rule_prior_crps | +0.07557% | 16/20 |
| 11 | empirical_constant | +0.37766% | 12/20 |
| 11 | gaussian_constant | +3.37247% | 18/20 |
| 22 | static_prior_nll | +1.59815% | 13/20 |
| 22 | rolling_rule_prior_nll | +1.45935% | 11/20 |
| 22 | rolling_neural_prior_nll | +1.55549% | 12/20 |
| 22 | static_prior_crps | +0.06389% | 14/20 |
| 22 | rolling_rule_prior_crps | +0.08187% | 17/20 |
| 22 | empirical_constant | -0.91693% | 8/20 |
| 22 | gaussian_constant | +2.11680% | 15/20 |
