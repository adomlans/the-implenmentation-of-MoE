# MoE 稀疏推理赛道 · 学习包

对标 TensorTonic 付费模块「MoE 稀疏推理」的免费自建练习。全部代码**纯 PyTorch CPU 可跑、自带断言自测、零外部依赖**，适合无 GPU 环境自学大模型系统知识。

## 一句话定位
> MoE = 用**显存**换算力与带宽，用**调度**换均衡，用**工程**兑现红利。

## 目录结构
```
moe/
├── task1_moe_basics.py        # 专家+Router+Top-k 路由（稀疏执行、参数与计算解耦）
├── task2_gating_balance.py    # Gating 与负载均衡（路由崩溃、Switch aux loss）
├── task3_expert_parallel.py   # 路由与 Token 调度（EP 搬运、容量约束、通信:计算比）
├── task4_mini_moe_lm.py       # 端到端 mini-MoE 语言模型（dense 对比，泛化反超）
├── task5_inference_cost.py    # 推理成本总账（四维账本 + 延迟实测）
├── README.md                  # 本文件
└── SUMMARY.md                 # 41 道思考题全解
```

## 运行
```bash
cd moe
python3 task1_moe_basics.py    # 期望输出 [PASS] Task1 全部断言通过
python3 task2_gating_balance.py
python3 task3_expert_parallel.py
python3 task4_mini_moe_lm.py
python3 task5_inference_cost.py
```
每个 Task 末尾都有 `[PASS]` 断言确认，失败即停止（自验闭环）。

## 5 Task 学习路径与实测结果

| Task | 主题 | 核心结论（本机实测） |
|---|---|---|
| 1 | 专家 + Router + Top-k | 32 token 路由分布 [15,16,19,14]；top-2 计算 50 MFLOP vs 全量 101（正好一半）；路由熵 1.380/1.386 |
| 2 | Gating 与负载均衡 | 无 aux：某专家垄断 84%（熵 0.59，CV 1.59）；有 aux：最大占比 0.34（熵 1.34，CV 0.34）；L_aux = 1.87 |
| 3 | 路由与 Token 调度 | 慢卡/理想 = 1.31x（straggler）；C=1.0 → 1.00x 均衡（丢 6 token）；真实 7B FFN 通信:计算 = 9.3e-5（EP 大赚） |
| 4 | 端到端 mini-MoE LM | 8 专家 top-2：FFN 参数 2x、FLOPs 0.5x；held-out 验证 5.731 vs dense 5.886（泛化反超 0.155）；路由熵全程 >2.05/2.08 |
| 5 | 推理成本总账 | 显存 1.46x（贵）/ 带宽 0.5x（省）/ FLOPs 0.5x（省）；decode 实测 2.19x 反而慢（朴素循环吃红利）、prefill 0.98x 持平——**红利兑现靠 GroupedGEMM** |

## 关键工程教训（踩坑记录）
1. **路由崩溃的"赢家"由初始化偶然性决定**（无 aux 时垄断者是专家 1 而非数据偏斜带火的专家 0）——崩溃本质是正反馈放大初始化偏差，与数据无关；
2. **总熵管"不崩溃"、条件熵管"真分工"**——路由熵高 ≠ 没分工（要看给定主题的条件分布）；
3. **账本减半 ≠ 实测减半**——decode 是小矩阵+调度主导，理论 FLOPs 红利被 kernel 开销吃光；真实系统必须 GroupedGEMM / persistent kernel；
4. **Switch 式配置 = 参数 2x + 计算 0.5x**（8 专家 top-2），不是"每专家变宽"；
5. **防背题**：训练/验证窗口必须来自不同文本半区，滑动窗口步长 1 时验证集天然泄露。

## 延伸
- 学完本赛道后建议：FlagOS Track1（SGLang 算子优化，Kaggle，报名至 11/13）直接实战 MoE kernel；MLSys FlashInfer AI Kernel 大赛论文（arXiv 2607.17979 / 2608.14560）看 Fused MoE 92.68x 怎么砍出来的。
- 依赖前置：建议先完成 quant（量化）赛道，Task5 的带宽账本与量化互证。
