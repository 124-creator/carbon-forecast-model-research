# 实际查新范围与创新边界

## 训练前实际获得的直接近邻

经正常GitHub API获得作者仓库StanfordASL/ALPaCA的固定commit、main/alpaca.py、README和LICENSE，源码仅留scratch，不作本项目依赖，也不在本次发布中转发。

- `StanfordASL/ALPaCA` commit `d06391a1bf11beda573078a6a1d62418aac367c4`, `main/alpaca.py`, blob `765bca1e89daeed1690e296d6bd86412f91b7796`, SHA256 `ed9706d84e88a4c117e7300af3063f72f305aafc651444ed7dd797561cf2c3bb`.
- `StanfordASL/ALPaCA` commit `d06391a1bf11beda573078a6a1d62418aac367c4`, `LICENSE`, blob `9382d991962006054f92bf86f4ec833aafb2479e`, SHA256 `6ecfeee423d0d9e21585810c3302765b25c009c4eb8bc86f5366884697ca5158`.
- `StanfordASL/ALPaCA` commit `d06391a1bf11beda573078a6a1d62418aac367c4`, `README.md`, blob `0564f4dcd3474a6c3efcf12fc06041cf12c73196`, SHA256 `71ea6dccc22e40bc49e6faf157600fce20c296426c29916e67bb96029b718735`.

训练前已静态核读main/alpaca.py中神经特征、先验精度、上下文贝叶斯/线性后验、预测均值与损失、对上下文后验求导和梯度优化特征的实现。训练后另读了AdaptiveDynamics的多步展开/不确定度传播部分，不冒充训练前先验。已核作者实现足以说明“学习神经特征+解析回归头+通过适配后的预测误差学习”是已有方法家族，不能以这三件事作为本轮原创。它不自动证明本轮每个期限成熟规则已被同一实现覆盖，也不认证本轮规则新颖。

## 搜索发现不等于方法核验

训练前实际搜索包括ALPaCA、变量投影循环网络、RLS元学习、在线多步预测延迟标签等；原始搜索返回留在scratch/carbon_stable_recurrent_direction_20261015，未发布第三方全文。LMS/RNN、DeltaNet/Longhorn等只作候选近邻发现，没有本轮原文方法核验。

训练进行中补查了在线预测+延迟反馈+元学习，检索到了Proactive Model Adaptation、TEFL和ADAPT-Z等标题/索引。它们不是本轮设计先验认证；没有把训练后的检索追认为训练前查新，也没有凭标题判定本轮机制不新或已新。

## 明确访问失败

本轮以下URL均由web_extract真实请求，全部返回`Blocked: URL targets a private or internal network address`，没有取得正文，也没有通过镜像、别名、IP或关闭安全检查绕行：

- `https://arxiv.org/abs/1807.08912`
- `https://arxiv.org/html/2410.11687v3`
- `https://arxiv.org/html/2205.12406v1`
- `https://arxiv.org/html/2602.22520v1`
- `https://arxiv.org/html/2412.08435v4`

后两次发生在训练启动后。另一次仓库HTML提取也被同一访问检查阻止；它没有作为本轮研究事实的认证。作者代码获取使用已有正常GitHub API的另一公开作者仓库StanfordASL/ALPaCA，不是上述论文正文的替代获取，也不冒称认证了论文全文。

## 对原M候选的查新边界（机制已在后续验分中失败）

M的半Markov/停留时长机制在后续固定版本独立验分中已经失败，不再作为当前候选。其作者访问声明与本方认证范围分开，旧版前序Stage L对不上四状态头的问题仍保留。不能依据本地字符串匹配或其他工作线自述，认证论文原文已取得或原创性已成立。详见相邻github_direction_followup_20261015记录。
