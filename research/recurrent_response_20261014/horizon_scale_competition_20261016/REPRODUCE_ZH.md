# 运行与复核入口

当前结论见REPORT_ZH.md。完整组合评分在supplement_20261017/RESULTS.json及VERIFY_NORMAL.json；原INCOMPLETE.json仍是原运行的有效状态。不要直接把supplement完成读成原1800秒运行成功。

环境复用本机独立venv：Python3.11.15，NumPy1.26.4，Torch2.3.1+cpu，float64，单线程数学库，CUDA关闭。依赖锁定见../fair_recurrent_tuning_20261016/requirements.lock.txt。本轮没有安装新依赖。

在本目录运行（python表示已选定的上述独立环境）：

    python -B -m unittest test_hsc test_hsc_supplement -v
    python -B -O -m unittest test_hsc test_hsc_supplement -v

对有本地保存预测的当前工作区：

    python -B hsc_verify.py --supplement --output NEW_VERIFY.json
    python -B -O hsc_verify.py --supplement --output NEW_VERIFY_O.json

评分输出用独占创建，禁止覆盖已有核验。无--supplement的原严格入口仍拒绝原不完整运行。--allow-partial只核对保存部分，不给不完整模型排名。

原始新实验执行器hsc_run.py只允许没有local_only的全新目录；当前目录拒绝覆盖。旧hsc_resume.py受原截止与INCOMPLETE限制，不能用于现在继续。补充执行器hsc_supplement.py依次init、batch、aggregate，只接受冻结的59个缺失任务；当前已经完成，禁止重复执行。BATCH起止记录、补充PRE_RUN、原始文件摘要保存全过程。

模型依赖同仓库../fair_recurrent_tuning_20261016/fair_core.py与fair_data.py，数据定位为仓库data/parsed/figshare_28682348/guangzhou_reported.csv的已核29204字节前缀。冻结摘要在PRE_RUN.json；不读取后续2022/2023记录。

公开包不含每个起点的预测、参数和运行输入缓冲；当前独立验分需要未推送的local_only。因此公开版本可以查看/运行单元测试和从原价重新训练，但不是下载后直接重放全部保存预测的完整运行包。重新训练为另一次运行，不能冒充原始已保存证据。训练日志汇总在supplement_20261017/TRAINING_LOGS.json，不含参数与逐点预测。

所有文献访问限制见SOURCE_SCOPE_CURRENT_ZH.md。没有作者模型运行复现、全文查新通过或发表资格认证；未调用子agent，代码自审不称独立代码审查。
