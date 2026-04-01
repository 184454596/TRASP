# Beyond Single-Transform Safety Preservation：面向大语言模型适配与剪枝的跨变换最坏情形安全鲁棒性（TRASP）

> 这是一份**面向实现的论文型说明文档**。  
> 目标不是追求发表版措辞，而是尽可能把 **问题定义、符号、方法、公式、训练流程、实验矩阵和实现细节** 写清楚，便于 Codex 直接据此实现代码与实验流水线。

---

## 0. 给 Codex 的阅读说明

本文档的核心目标是实现并验证如下论文主张：

> **single-transform safety preservation 并不蕴含 cross-transform worst-case robustness。**

更具体地说：

- 现有很多方法只在**单一 post-training 操作**下保持安全，例如：
  - 只对 fine-tuning 后安全退化做修复；
  - 只对 pruning 后安全问题做修复；
- 但现实部署中，开放权重模型常常会经历**多种不同的后处理变换**；
- 本文关心的对象是：
  - **现实的 benign / realistic post-deployment adaptation**
  - **deployment-time pruning**
  - 它们共同构成的 **cross-transform worst-case safety robustness**

本文档中的主方法为：

- **TRASP** = hard-transform mining + transformed-model safety projection + variance stability
- **TRASP+** = TRASP + representation consistency（可选增强版，不是主方法必需项）

本文档中的**主 threat model**是：

- **benign / realistic post-deployment adaptation + pruning**
- **不是**把 malicious fine-tuning 作为主设定  
- harmful fine-tuning 文献只作为：
  - 问题动机来源
  - baseline 对比来源
  - 方法设计借鉴来源

---

## 1. 摘要

对齐后的大语言模型通常被默认在部署后仍然安全，但这一假设在开放权重场景中并不成立。模型在发布后往往会经历现实且高频的参数层变换，例如面向任务迁移的小步 LoRA 适配，以及面向本地部署的 pruning 压缩。已有研究分别表明：下游 fine-tuning 会显著削弱安全对齐，而剪枝本身也会暴露 deployment-time security gap。然而，现有方法大多仅针对**单一 post-training 操作**做安全保持或事后修复，尚未将 **adaptation family 与 pruning family 上的跨变换最坏情形安全** 作为统一训练目标。

本文提出 **TRASP**，研究 aligned LLM 在现实部署后 **adaptation + pruning** 两类算子上的 **worst-case cross-transform safety robustness**。我们的核心判断是：

> **single-transform safety preservation 并不蕴含 cross-transform worst-case robustness。**

围绕这一判断，本文将 post-deployment 安全形式化为跨变换最坏情形约束优化问题，并提出一个 transformation-aware 训练框架 TRASP。TRASP 由三个组件组成：

1. **hard-transform mining**：在训练中显式近似当前最危险的部署变换；
2. **transformed-model safety projection**：把投影式安全约束从静态模型扩展到 hardest transformed model；
3. **cross-transform stability**：用风险方差约束缩窄 transform-wise 风险分布，并可选加入表示一致性增强项。

对于离散 pruning 算子，TRASP 在真实前向与 STE 反向组成的局部 surrogate 上执行投影，因此我们将其视为近似优化机制，而非对原始离散目标的精确可行性保证。

实验设计围绕三个问题展开：

1. 单一变换上的安全保持方法是否会在 cross-transform worst-case 评测下失效；
2. TRASP 是否能在已见 transforms 上降低 worst-case ASR，同时保持 utility 与低过拒绝；
3. 这种鲁棒性是否能泛化到未见的 LoRA rank、步数、剪枝算法与稀疏率，并在顺序组合变换（LoRA→Prune）下保持优势。

---

## 2. 问题背景与论文定位

### 2.1 现有工作的三条主线

当前最相关的研究大体分成三类：

1. **post-fine-tuning safety preservation / repair**
   - 例如 EnchTable、SPARD、Safe Delta、Antidote、SafeGrad、Booster、Learning to Stay Safe
   - 关注：fine-tuning 之后如何保持或修复安全

2. **deployment-time pruning attack / repair**
   - 例如 Fewer Weights, More Problems、SPLoRA、DDI-Pruning 一类工作
   - 关注：pruning 是否会触发或修复安全问题

3. **inference-time defense**
   - 例如 ADA
   - 关注：不改变模型参数，在推理时通过 token / header / latent intervention 恢复安全

### 2.2 本文和现有工作的边界

本文**不**声称统一所有安全问题，也**不**试图替代 inference-time defense。

本文的对象是：

> **对 post-deployment adaptation 与 pruning 两类现实变换的跨变换最坏情形安全鲁棒性。**

本文最关键的 claim 是：

> 现有 single-transform preservation 方法，在其目标变换上有效，  
> 但这**不意味着**它们在其他现实部署变换下仍然安全。

因此本文要证明的是一个新的对象：

- 不是 static safety
- 不是 single-transform preservation
- 而是 **cross-transform worst-case robustness**

---

## 3. 符号表与形式化对象

### 3.1 基本符号

- $\theta$：已对齐基础模型的参数
- $f_\theta$：由参数 $\theta$ 定义的语言模型
- $x^-$：有害输入（harmful prompt）
- $x^+$：良性输入（benign prompt）
- $D^-$：有害输入分布
- $D^+$：良性输入分布
- $\mathcal{L}_{task}$：任务效用损失
- $R_T(\theta)$：最终 judge-based 安全风险
- $\tilde r_T(\theta)$：训练 surrogate 风险
- $\tau$：允许的 worst-case 风险阈值

### 3.2 变换族定义

本文只研究两类 transform family。

#### （1）Adaptation transforms

这是现实部署中的 benign / realistic 小步适配，不是恶意 fine-tuning。

定义为：

$$
T_{\text{adapt}}(\theta;\xi)=\theta+U_k(\theta;\xi)
$$

其中：

- $\xi$：adaptation 配置
  - LoRA rank
  - learning rate
  - step count
  - adapter target modules
  - benign downstream domain
- $U_k$：由 $k$ 步小更新得到的参数漂移

#### （2）Pruning transforms

这是部署前的压缩或稀疏化操作。

定义为：

$$
T_{\text{prune}}(\theta;\zeta)=M_\zeta \odot \theta
$$

其中：

- $\zeta$：pruning 配置
  - algorithm（Magnitude / Wanda / SparseGPT）
  - sparsity ratio
  - optional structure type
- $M_\zeta \in \{0,1\}^{|\theta|}$：pruning mask

### 3.3 训练 / 测试变换族

训练时只暴露一部分 seen transforms：

$$
\mathcal T_{\text{train}}
=
\mathcal T_{\text{adapt}}^{\text{seen}}
\cup
\mathcal T_{\text{prune}}^{\text{seen}}
$$

测试时加入 unseen transforms 与顺序组合 stress test：

$$
\mathcal T_{\text{test}}
=
\mathcal T_{\text{train}}
\cup
\mathcal T_{\text{unseen}}
\cup
\mathcal T_{\text{compose}}
$$

其中：

- $\mathcal T_{\text{unseen}}$：未见超参数与未见算法
- $\mathcal T_{\text{compose}}$：例如 LoRA → Prune 的顺序组合

---

## 4. 论文目标：从静态安全到跨变换最坏情形安全

传统安全对齐关注的是静态模型风险：

$$
\min_\theta \ \mathcal L_{\text{task}}(\theta)+\lambda \mathcal L_{\text{safe}}(\theta)
$$

这隐含了一个强假设：部署时模型参数仍然是 $\theta$。

本文认为真正应被约束的是**变换后的最坏情形风险**：

$$
\min_\theta \ \mathcal L_{\text{task}}(\theta)
\quad
\text{s.t.}\quad
\max_{T\in \mathcal T_{\text{train}}} R_T(\theta)\le \tau
$$

其中：

$$
R_T(\theta)
=
\mathbb E_{x\sim D^-}
\left[
\mathbf 1\big(f_{T(\theta)}(x)\text{ unsafe}\big)
\right]
$$

这就是本文的核心对象：

> **worst-case cross-transform safety robustness**

---

## 5. 训练 surrogate：为什么需要 surrogate，如何定义 surrogate

最终的 judge-based ASR / harmfulness risk 无法直接求梯度，因此训练必须使用 surrogate 风险。

### 5.1 Harmful refusal margin

对有害输入 $x^- \sim D^-$，给定：

- $y_{\text{refuse}}$：拒答目标
- $y_{\text{unsafe}}$：危险续写模板

定义拒答边际：

$$
\Delta^-_T(x^-)
=
\log p_{T(\theta)}(y_{\text{refuse}}\mid x^-)
-
\log p_{T(\theta)}(y_{\text{unsafe}}\mid x^-)
$$

定义有害风险 surrogate：

$$
\mathcal L^-_T
=
\mathbb E_{x^- \sim D^-}
\left[
\max(0,m^- - \Delta^-_T(x^-))
\right]
$$

其中 $m^-$ 是 refusal margin 的阈值。

### 5.2 Benign non-refusal margin

对良性输入 $x^+ \sim D^+$，给定：

- $y_{\text{help}}$：期望帮助性回复
- $y_{\text{refuse}}$：拒答模板

定义帮助性边际：

$$
\Delta^+_T(x^+)
=
\log p_{T(\theta)}(y_{\text{help}}\mid x^+)
-
\log p_{T(\theta)}(y_{\text{refuse}}\mid x^+)
$$

定义 benign non-refusal surrogate：

$$
\mathcal L^+_T
=
\mathbb E_{x^+ \sim D^+}
\left[
\max(0,m^+ - \Delta^+_T(x^+))
\right]
$$

### 5.3 总 surrogate 风险

定义：

$$
\tilde r_T(\theta)
=
\mathcal L^-_T
+
\lambda_b \mathcal L^+_T
$$

注意：

- $R_T(\theta)$：最终 judge-based 风险
- $\tilde r_T(\theta)$：训练 surrogate 风险

二者不是同一指标，因此实验中必须额外做：

- surrogate–judge calibration
- Spearman correlation
- risk binning / calibration curve
- optional AUC

这样才能说明 surrogate 是 judge-based 风险的有效 proxy。

---

## 6. 方法：TRASP

TRASP 的核心由三部分构成：

1. hard-transform mining
2. transformed-model safety projection
3. cross-transform stability

默认主方法为：

$$
\text{TRASP}=
\text{hard-transform mining}
+
\text{transformed-model projection}
+
\mathcal L_{\text{var}}
$$

增强版为：

$$
\text{TRASP+}
=
\text{TRASP}
+
\beta_2 \mathcal L_{\text{repr}}
$$

---

## 7. Hard-transform mining

### 7.1 动机

如果直接优化：

$$
\max_{T\in\mathcal T_{\text{train}}}\tilde r_T(\theta)
$$

每步都要枚举所有 transform，成本太高。

如果改为平均风险：

$$
\mathbb E_{T\sim\mathcal T_{\text{train}}}\tilde r_T(\theta)
$$

则会退化成 naive multi-transform ERM，而这正是论文最强的替代解释。

因此 TRASP 采用折中做法：

- 每步只看一个小候选集 $\mathcal S_t$
- 在其中挑 hardest transform

### 7.2 定义

令：

$$
\mathcal S_t = \{T_1,\dots,T_M\}\subset \mathcal T_{\text{train}}
$$

先做 utility proposal，得到 proposal point $\theta^+$（后文定义）。然后在 proposal point 上选 hardest transform：

$$
T_t^\star
=
\arg\max_{T\in \mathcal S_t}\tilde r_T(\theta^+)
$$

### 7.3 工程实现：cheap-score → full-backward

为适配 4×A6000 的预算，推荐实现两阶段 mining：

1. 对每个候选 transform 做 **cheap score**
   - 小 batch
   - 只前向
   - 或近似 surrogate 估计
2. 选 top-1 或 top-2
3. 仅对 hardest transform 做 full backward

### 7.4 作用

这一步是 TRASP 与 naive ERM 的第一层本质差异：

- naive ERM：平均 transformed risk
- TRASP：当前最危险 transform 的局部最坏情形风险

---

## 8. Transformed-model safety projection

这是 TRASP 的核心。

### 8.1 设计思想

借鉴 SPAG / SPARD 的几何思路：

1. 先做任务优化 proposal
2. 再在局部安全约束下投影回可行区域

但与 SPAG 不同的是：

- SPAG 约束的是当前模型的安全
- TRASP 约束的是 **hardest transformed model** 的安全 surrogate

### 8.2 Step 1：utility proposal

给定任务损失 $\mathcal L_{\text{task}}$，先做一步标准任务更新：

$$
\theta^+ = \theta - \eta \nabla_\theta \mathcal L_{\text{task}}(\theta)
$$

这里的 $\theta^+$ 是 **proposal point**。

### 8.3 Step 2：选择 hardest transform

在 $\theta^+$ 处选择 hardest transform：

$$
T_t^\star = \arg\max_{T\in \mathcal S_t}\tilde r_T(\theta^+)
$$

### 8.4 Step 3：局部线性化

计算 hardest transform 的 surrogate 风险梯度：

$$
g_t^\star = \nabla_\theta \tilde r_{T_t^\star}(\theta^+)
$$

在 $\theta^+$ 附近做局部一阶线性化，定义局部安全半空间：

$$
\mathcal C^+
=
\left\{
\theta' :
\tilde r_{T_t^\star}(\theta^+)
+
\langle g_t^\star,\theta'-\theta^+ \rangle
\le \tau
\right\}
$$

### 8.5 Step 4：投影

在该半空间内找最接近 proposal point 的点：

$$
\theta^{new}
=
\arg\min_{\theta'\in\mathcal C^+}
\|\theta'-\theta^+\|_2^2
$$

闭式解：

$$
\theta^{new}
=
\begin{cases}
\theta^+, & \tilde r_{T_t^\star}(\theta^+)\le \tau,\\[4pt]
\theta^+ - \alpha_t g_t^\star, & \text{otherwise},
\end{cases}
$$

其中

$$
\alpha_t
=
\min\left(
\frac{\tilde r_{T_t^\star}(\theta^+)-\tau}{\|g_t^\star\|_2^2},
\eta_{\text{safe}}
\right)
$$

$\eta_{\text{safe}}$ 是 trust-region cap，避免 correction 过大导致 utility 明显下降。

最终更新：

$$
\theta \leftarrow \theta^{new}
$$

### 8.6 这一步的含义

TRASP 的更新不是“再加一个 transformed loss”，而是：

> 每一步 task update 都必须对当前 hardest transformed model 的局部安全约束负责。

---

## 9. Smooth adaptation 与 discrete pruning：为什么必须分开讲

这是实现时最容易出错，也是论文里最需要说清楚的部分。

### 9.1 Adaptation transforms：smooth case

对于 LoRA / continued SFT 这种小步 adaptation：

- transform 本身是可微的
- surrogate 风险关于 $\theta$ 具有较好的局部光滑性

因此 transformed projection 仍保留类似 SPAG 的局部一阶解释。

### 9.2 Pruning transforms：discrete case

对于 pruning：

- mask 是离散的
- 真实目标不是光滑函数
- 不存在和 smooth case 完全相同的梯度语义

因此本文采用：

- **前向**：真实 pruning mask
- **反向**：STE surrogate

也就是说，优化时使用的是：

$$
\tilde r_{T_{\text{prune}}}^{\text{STE}}(\theta)
$$

而不是原始离散目标的精确梯度。

### 9.3 必须在论文和代码里写清楚的话

对于 pruning case，TRASP 提供的是：

> **STE-based local surrogate optimization**

而不是：

> 对原始离散 transformed risk 的精确一阶 feasibility guarantee

这句话在代码注释、README 和论文里都要写清楚。

### 9.4 建议的验证实验

在小模型、小样本上做：

- STE gradient direction vs finite-difference direction agreement
- 看方向一致率 / cosine similarity
- 不要求严格理论正确，只验证经验可用性

---

## 10. Cross-transform stability

最新版建议把它做成“主模块 + 可选增强模块”，避免方法过重。

### 10.1 主模块：风险方差约束

定义：

$$
\mathcal L_{\text{var}}
=
\operatorname{Var}_{T\sim\mathcal T_{\text{train}}}
[\tilde r_T(\theta)]
$$

作用：

- 缩窄 transform-wise 风险分布
- 避免有些 transforms 很安全、有些 transforms 很危险

### 10.2 可选增强：表示一致性项

定义某个安全相关中间表示 $h_T(x)$，例如：

- final-layer assistant token hidden state
- pooled representation
- safety probe feature

定义：

$$
\mathcal L_{\text{repr}}
=
\mathbb E_x
\sum_{T_i,T_j\in\mathcal S_t}
\|h_{T_i}(x)-h_{T_j}(x)\|_2^2
$$

### 10.3 为什么降级为可选模块

这一项的问题是：

- 训练开销明显增加
- 表示定义自由度大
- 可能导致方法显得过重

因此主方法只用：

$$
\mathcal L_{\text{var}}
$$

增强版再加：

$$
\mathcal L_{\text{repr}}
$$

这样：

- **TRASP**：主论文方法
- **TRASP+**：增强版

---

## 11. 总训练目标

一个实现友好的写法如下。

### 11.1 主优化逻辑

每一步：

1. 计算 task proposal
2. 在 proposal point 挑 hardest transform
3. 对 hardest transform 的 surrogate 做局部投影
4. 可选加入 variance stability 的估计与日志

### 11.2 写成约束优化

论文形式：

$$
\min_\theta \ \mathcal L_{\text{task}}(\theta) + \beta_1 \mathcal L_{\text{var}}(\theta)
\quad
\text{s.t.}\quad
\max_{T\in\mathcal T_{\text{train}}}\tilde r_T(\theta)\le \tau
$$

增强版：

$$
\min_\theta \ \mathcal L_{\text{task}}(\theta)
+ \beta_1 \mathcal L_{\text{var}}(\theta)
+ \beta_2 \mathcal L_{\text{repr}}(\theta)
\quad
\text{s.t.}\quad
\max_{T\in\mathcal T_{\text{train}}}\tilde r_T(\theta)\le \tau
$$

### 11.3 和 naive multi-transform ERM 的区别

naive ERM：

$$
\mathcal L_{\text{task}}(\theta)
+
\lambda
\mathbb E_{T\sim\mathcal T_{\text{train}}}\tilde r_T(\theta)
$$

区别：

1. ERM 优化平均 transformed risk  
2. TRASP 对 hardest transform 做局部最坏情形投影  
3. ERM 没有几何约束结构，TRASP 有显式可行性修正  
4. ERM 更像 parameter-space augmentation，TRASP 更像 worst-case constrained optimization

---

## 12. 算法伪代码（适合 Codex 实现）

```text
Algorithm: TRASP

Input:
- aligned base model parameters θ
- task batches B_task
- harmful data D-
- benign data D+
- seen transform family T_train
- threshold τ
- trust-region cap η_safe

for each training step t:
    1. sample task batch B_task
    2. compute utility proposal:
           θ_plus = θ - η * grad L_task(θ)

    3. build candidate transform subset S_t ⊂ T_train

    4. cheap-score each transform in S_t at θ_plus
       keep top-k hardest candidates

    5. fully evaluate surrogate risk on top-k candidates
       choose:
           T_star = argmax_T r_tilde_T(θ_plus)

    6. compute transformed surrogate gradient:
           g_star = grad r_tilde_{T_star}(θ_plus)

    7. if r_tilde_{T_star}(θ_plus) <= τ:
           θ_new = θ_plus
       else:
           alpha = min((r_tilde_{T_star}(θ_plus)-τ) / ||g_star||^2, η_safe)
           θ_new = θ_plus - alpha * g_star

    8. update θ ← θ_new

    9. log:
       - T_star
       - transform-wise surrogate risks
       - projection step size
       - trust-region clipping
       - optional variance stability stats

return θ
```

---

## 13. 实验设计总览

### 13.1 核心研究问题

#### RQ1
**single-transform preservation 是否会在 cross-transform worst-case 评测下失效？**

#### RQ2
**TRASP 是否能在 seen transforms 上降低 worst-case ASR，同时保持 utility 与低过拒绝？**

#### RQ3
**TRASP 是否能泛化到 unseen transforms，并在顺序组合变换下保持优势？**

### 13.2 主设定必须保持不变

主设定是：

- benign / realistic post-deployment adaptation
- deployment pruning

不是：

- malicious fine-tuning as main setting

---

## 14. 训练与测试变换矩阵

### 14.1 Seen transforms

训练时建议只用四个：

#### Adaptation
- LoRA rank 8, 3-step
- LoRA rank 16, 3-step

#### Pruning
- Magnitude pruning, 20%
- Wanda pruning, 20%

### 14.2 Unseen transforms

#### Unseen adaptation
- LoRA rank 32, 3-step
- LoRA rank 16, 5-step
- LoRA rank 8 with different LR
- benign held-out domain continued SFT

#### Unseen pruning
- Magnitude 50%
- Wanda 50%
- SparseGPT 20%
- 2:4 semi-structured pruning

### 14.3 Sequential composition stress test

只做评测，不默认纳入训练：

- LoRA(rank 8, 3-step) → Magnitude 20%
- LoRA(rank 16, 3-step) → Wanda 20%
- held-out adaptation → held-out pruning

---

## 15. 数据与指标

### 15.1 数据类型

- harmful_train / harmful_val / harmful_test
- benign_eval
- utility_eval
- composition_eval

### 15.2 主指标

主指标必须是：

$$
\text{Worst-Case ASR}
=
\max_{T\in\mathcal T_{\text{eval}}}
\text{ASR}(T(\theta))
$$

### 15.3 其他指标

- Mean ASR across transforms
- Transform-wise variance
- Utility accuracy / score
- Benign refusal
- Harmfulness score（若 judge 输出标量）
- Training hours
- Peak GPU memory
- Transform mining overhead

### 15.4 诊断性指标

- surrogate–judge Spearman correlation
- calibration curve / risk bins
- STE direction agreement
- safety–utility Pareto frontier

---

## 16. 主 baseline 设计

### 16.1 第一层：下界

- Base aligned model
- Task-only adaptation

### 16.2 第二层：最强替代解释

- Naive multi-transform ERM

### 16.3 第三层：single-transform preservation / repair baselines

主表建议至少包含：

- ENCHTABLE
- SPARD / SPAG
- Safe Delta
- Antidote
- SafeGrad 或 Booster（至少一个）

### 16.4 额外近邻方法

可放附录或补充表：

- SPLoRA
- Learning to Stay Safe
- DDI-Pruning
- 其他较新 post-fine-tuning safety regularization 方法

### 16.5 ADA 的位置

ADA 不应放进主 baseline battle。更合适的是单独做：

- Base
- Base + ADA
- TRASP
- TRASP + ADA

目的不是打败 ADA，而是说明：

- TRASP：参数层 robustness
- ADA：推理时 defense
- 两者互补

---

## 17. 主表结构

### 表 1：single-transform preservation 不等于 cross-transform robustness

列：

- Seen-Adapt ASR
- Seen-Prune ASR
- Cross-Transform Worst-Case ASR
- Utility
- Benign Refusal

这张表是全文灵魂。

### 表 2：seen transforms 上的主结果

列：

- Worst-Case ASR
- Mean ASR
- Transform Variance
- Utility
- Benign Refusal

### 表 3：unseen-transform generalization

列：

- Unseen Adapt Hyperparams
- Unseen Prune Hyperparams
- Unseen Prune Algorithm
- Overall Worst-Case
- Utility

### 表 4：sequential composition stress test

列：

- LoRA→Magnitude ASR
- LoRA→Wanda ASR
- Held-out Compose Worst-Case
- Utility

### 表 5：核心消融

行：

- TRASP
- w/o hard-transform mining
- w/o transformed projection
- w/o L_var
- TRASP+
- naive multi-transform ERM

列：

- Worst-Case ASR
- Mean ASR
- Variance
- Utility
- Training Cost

### 表 6：工程成本

列：

- Avg extra forwards / step
- Avg full backwards / step
- Peak GPU memory
- Training hours
- Worst-Case ASR
- Utility

### 表 7：与 ADA 的交互

列：

- Deep-Prefill ASR
- Cross-Transform Worst-Case ASR
- Utility
- Benign Refusal
- Latency

---

## 18. 主图结构

### 图 1：transform-wise ASR 分布

推荐 boxplot / violin plot。

目的：

- 展示最坏情况变低
- 展示 transform-wise 风险分布缩窄

### 图 2：Safety–Utility Pareto frontier

横轴 utility，纵轴 worst-case ASR。

通过扫描：

- $\tau$
- $\eta_{\text{safe}}$

来画曲线。

### 图 3：surrogate–judge calibration

横轴 surrogate risk，纵轴 empirical ASR / HS。

作用：

- 说明 surrogate 不是瞎优化
- 说明它能排序 transformed risk

---

## 19. 结果写作逻辑模板

### 第一段：RQ1

现有 single-transform preservation / repair 方法在其目标 operator 上表现有效，但在 cross-transform worst-case 评测下显著退化，说明 operator-specific 成功并不等价于 deployment robustness。

### 第二段：RQ2

TRASP 在 seen transforms 上显著降低 worst-case ASR 和 transform variance，同时在 utility 与 benign refusal 上保持更优平衡，证明 hardest-transform-aware projection 是有效的。

### 第三段：RQ3

TRASP 的优势延续到 unseen transforms，包括未见 rank、步数、算法和稀疏率，表明它学到的是更稳定的 post-transform safety property，而不是 seen-transform memorization。

### 第四段：补充证据

- surrogate–judge calibration 支持 surrogate 的合理性
- STE direction agreement 支持 pruning case 的经验可用性
- Pareto frontier 支持 worst-case 优化没有带来不可接受的过保守

---

## 20. 实现要点（给 Codex 的工程提示）

### 20.1 必须记录的日志

每次训练与评测都要保存：

- full config
- git commit hash
- random seed
- package versions
- PyTorch/CUDA info
- hostname
- timestamp

### 20.2 TRASP 每步建议记录

- selected hardest transform
- candidate transform scores
- full surrogate risks
- projection step size
- trust-region clipping events
- transform family statistics
- pruning case STE flags

### 20.3 统一输出格式

所有 baseline 和 TRASP 都必须输出同一种 machine-readable 格式：

- `metrics.json`
- `metrics.csv`
- `config.yaml`
- `env.json`

否则后续聚合会非常痛苦。

### 20.4 优先级排序

先实现：

1. data + manifests
2. transform abstraction
3. eval stack
4. Base / Task-only / Naive ERM
5. TRASP
6. pilot sanity run
7. single-transform baselines
8. RQ1 / RQ2 / RQ3
9. sequential composition
10. ADA interaction
11. paper tables / figures

---

## 21. 论文局限性（也要让 Codex 知道）

1. 主文只覆盖 adaptation + pruning，不包含 merge、quantization、general editing。  
2. pruning case 是 STE-based local surrogate，不是离散目标的精确理论保证。  
3. 当前主验证规模是 3B / 7B text-only models。  
4. ADA 等 inference-time defense 是互补层，不是本文要替代的对象。  
5. sequential composition 目前只做 stress test，不是主训练目标。

---

## 22. 给 Codex 的“实现目标总结”

如果你是 Codex，实现本文时应始终记住：

### 主 claim
**single-transform safety preservation does not imply cross-transform worst-case robustness**

### 主 threat model
**benign / realistic post-deployment adaptation + pruning**

### 主方法
**TRASP = hard-transform mining + transformed-model projection + variance stability**

### pruning 语义
**STE-based local surrogate only**

### 主指标
**worst-case ASR across transforms**

### 最重要的三张表
- 表 1：single-transform preservation 失效
- 表 2：seen transforms 主结果
- 表 3：unseen-transform generalization

如果这三张表站不住，这篇论文就站不住。

---

## 23. 参考文献（实现导向）

> 这里只保留和实现最相关、最应该在代码仓库注释中被提及的工作。

1. **EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models**
2. **SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance–Diversity Data Selection**
3. **Fewer Weights, More Problems: A Practical Attack on LLM Pruning**
4. **Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth**
5. **Safe Delta / Antidote / SafeGrad / Booster**
6. **SPLoRA / DDI-Pruning**
7. **Learning to Stay Safe**

---

## 24. 一句话总结

这篇论文真正要做成的不是：

- 又一个 fine-tuning defense
- 又一个 pruning repair
- 又一个 inference-time guardrail

而是：

> **把 aligned LLM 的安全对象从单一静态 checkpoint，推进为对现实 post-deployment adaptation 与 pruning 的跨变换最坏情形鲁棒性。**
