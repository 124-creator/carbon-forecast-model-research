# PyTorch 分段底座执行结果

本轮建立并实际运行成熟 nn.GRU + AdamW，不主张基础框架创新。没有执行市场训练，没有读取市场CSV或2022价格/边界，没有下载模型权重，没有提交或推送。

## 环境与安装
项目独立环境：本目录 `.venv/Scripts/python.exe`；Python 3.11.15，torch 2.3.1+cpu，numpy 1.26.4；完整依赖见 requirements.lock.txt。include-system-site-packages=false。Hermes环境只作为创建venv的启动解释器，没有向其中安装包。机器12逻辑CPU，开始时E盘剩余156533067776字节。

首次 `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir --timeout 30 --retries 1` 下载619.4MB轮子到168.3MB时读取超时，失败。随后项目解释器执行 `-m pip install torch==2.3.1 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir --timeout 25 --retries 0` 成功；包含MKL约228.5MB，未超过预设1.5GB下载预算。真实官方网页已获取并保存OFFICIAL_INSTALL_RECEIPT.json；必须说明：官方文档核对发生在安装之后，未满足用户要求的“先核官方命令”顺序。没有修改证书验证，也没有绕过安全拒绝。

## 来源与许可
读取已取得的作者SegRNN模型与完整Apache-2.0许可，保留LICENSE。独立紧凑移植对应作者仓库lss-1138/SegRNN，既有审计固定commit 8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f；不是完整作者模型复现：仅价格、60输入、5点段长、16隐藏维度、两个稀疏解码查询提取1/5/20目标；取消RevIN、channel ID、整文件loader。backbone.py显式写明修改。

## 映射与梯度
映射原NumPy底座参数，wi/wh/emb/out转置；GRU门顺序r,z,n，分离输入/隐藏bias，候选门reset作用在隐藏仿射之后。全部参数逐坐标NumPy中心差分对比PyTorch反向梯度（不是仅单方向）。forward最大误差2.220446049250313e-16；梯度最大误差9.046413618207794e-11。测试先运行并因缺失底座失败，随后通过。NumPy只用于映射核验，不用于训练。

## 固定合成协议与结果
沿用已有384样本生成器、种子7/17、有信号abs(z)目标和无信号任务。时间顺序前192训练、随后64验证、末128测试；尺度只用前192拟合。AdamW lr=.003、weight_decay=.001、6批/epoch，最多200epoch，valid min_delta=1e-5、patience=25，clip=1，恢复验证最优状态。验证patience触发表示选模停止，不表示优化失败，也不要求stationarity。每批梯度范数、每epoch训练/验证误差、更新次数完整保存。

|任务|种子|测试MSE|零预测MSE|最优epoch/执行epoch|
|---|---:|---:|---:|---:|
|nonlinear|7|0.010749283900702794|0.1705294677598536|80/105|
|nonlinear|17|0.01172942009718815|0.1509063059977474|63/88|
|null|7|0.010258847792397932|0.010213159802830179|4/29|
|null|17|0.011110806857376601|0.010992236312385268|3/28|

信号门为MSE<0.5*零预测MSE；无信号门为MSE<=1.1*零预测MSE。四项通过。训练与映射总计18.922秒（不含安装和额外映射测试）。该结果说明此模拟器底座可学、无信号没有明显测试过拟合，不说明市场增益，更不以逐种子微小胜过旧实现作为底座门槛。训练样本从旧协议256改成192+64验证，因此不能直接把绝对分数差解释成框架优势。

## 前缀与下一轮451市场设计（未执行）
prefix_loader.py仅按既有审计记录数、非缓冲逐字节读入header+指定记录；不会请求下一条边界。用玩具CSV加非法UTF-8后缀实际核验，前缀仍正确读取。本轮真实市场没有打开。

沿用此前已审计广州author-reported序列950条、末日2021-12-31（读取数951含header），训练边界end=459；60输入、h=1/5/20、原固定451发行起点、2020/2021年内完整标签同口径，不更换/挑选市场。拟合尺度仅初始训练段；各h使用成熟标签。初始训练内部按时间预留验证段并purge20目标；固定单次初始拟合、不刷新，必须同时保留同支持成熟标签线性ridge和随机游走。市场尚未写执行器，不能声称已满足强滚动基线对比；若升级rolling40，须另行冻结同成本刷新与成熟支持。运行前核验451与既有origins一致，市场结果不能因合成通过而预判。

## 父级验证命令
在本目录，使用 `.venv/Scripts/python.exe test_mapping.py`、`.venv/Scripts/python.exe verify.py`、`.venv/Scripts/python.exe -O verify.py`；重训 `.venv/Scripts/python.exe run.py` 会覆盖本轮结果，先备份。保存分数正常/优化模式四项重算，最大误差0；这是保存结果重算，不是独立重训。
