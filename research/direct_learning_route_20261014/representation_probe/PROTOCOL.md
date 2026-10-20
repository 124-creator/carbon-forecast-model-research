# 直接树输入适配固定对照

只检验信息表示，非新算法。父read真实state/run.py，原状态g=过去60水平偏离归一tanh，v=20/120创新波动比。候选compact只past信息：g/v，最近1/5/20差分平均、60差分sd，共6维；不用MLE预测、残差或新外部信息。raw60为60差分。两者信息窗口不同（compact含120sd），故加入raw120对照，不把额外历史信息称表示算法增益。

同HGBT固定100树、lr.05、maxleaves7/minleaf40/l2.1/无随机验证，直接h1/5/20原价变化。共同训练起点>=120，窗口最多480、完整h20成熟、459初始化40刷新、451sameyear评分。每model12fit，全部报告，未在结果后改容量。compact少特征不称算力匹配。

数学future扰动两个模型训练/推理真实refit至少1代表不读取未来边界；prefix定长950，不2022price。相同历史120输入：比较compact/raw120；raw60仅原baseline接口诊断。无score择配置。此结果最多支撑表示方向，不是创新/独立test。

新来源获取：web_extract官方两个目标安全拒绝无重试；Google browser启动时pypi包TLS失败，无新source。不声明查新通过。15min运行上限，no commitpush。
