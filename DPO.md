## 1.概述
DPO（Direct Preference Optimization，直接偏好优化）是一种专门为LLMs设计的用“人类偏好数据”（成对比较：A 比 B 更好）来训练语言模型/策略的办法。它的卖点是：不需要显式训练奖励模型（RM），也不需要像 PPO 那样做复杂的在线强化学习；而是把问题改写成一个类似二分类/对比学习的监督目标，直接把策略往“更受偏好”的方向推。

## 2.想要理解DPO的推导过程，你必须要知道的预备知识
### 2.1 Kullback-Leibler 散度（KL 散度）
Kullback-Leibler 散度（KL 散度），又称为相对熵，是信息论中的一个重要概念。它用于衡量两个概率分布之间的差异，可以理解为两种分布之间的**信息差异**。

#### 2.1.1 KL散度定义
$$
D_{\mathrm{KL}}(P \| Q) = \sum_{x} P(x)\,\log\frac{P(x)}{Q(x)}
$$
(直觉上，他是我们是衡量用一个分布 $Q$ 去编码另一个目标分布 $P$ 时所需要的额外信息量。

#### 2.1.2 KL散度的性质

* （1）**非负性**：$D_{\mathrm{KL}}(P \| Q) \ge 0$，且当且仅当 $P(x) = Q(x)$ 时，$D_{\mathrm{KL}}(P \| Q) = 0$。这意味着当且仅当两个分布完全一致时，KL 散度为零。（<span style="color:red;">请记住这点，之后我们会用到</span>）
* (2) **非对称性**：$D_{\mathrm{KL}}(P \| Q) = D_{\mathrm{KL}}(Q \| P)$不一定成立

### 2.2 Bradley-Terry 模型 (BT)
Bradley-Terry 模型主要用于评估不同项目之间的相对强度或偏好。这种模型在体育比赛预测、产品推荐系统、社会科学中的偏好排序等多种领域都有广泛应用。

### 2.2.1 BT模型定义
假设我们有一组对象 $O_1, O_2, \ldots, O_n$，并且对于任意两个对象 $O_i$ 和 $O_j$，我们知道 $O_i$ 对 $O_j$ 获胜的概率。Bradley-Terry 模型的目标是通过一系列配对比较的结果来估计每个对象的相对强度。

Bradley-Terry 模型的核心假设是每个对象 $O_i$ 都有一个潜在的强度参数 $\lambda_i$，这个参数越大，该对象越强。对于任意两个对象 $O_i$ 和 $O_j$，$O_i$ 对 $O_j$ 获胜的概率 $P(i > j)$ 可以表示为：

$$P(i > j) = \dfrac{\lambda_i}{\lambda_i + \lambda_j}$$

这意味着，对象 $O_i$ 对 $O_j$ 获胜的概率是 $O_i$ 的强度除以两个对象强度之和。

2.2.2 强度参数估计
学过概率论的可能会觉得，对象 $O_i$ 与 对象 $O_j$的比赛胜负看起来有点像标准的伯努利分布。事实上也确实如此：BT（Bradley–Terry）模型把 $O_i$ 和 $O_j$ 比赛的胜负看成一个伯努利随机变量，并假设不同场次（或不同对局记录）在给定参数后是条件独立的。于是我们自然可以使用极大似然估计（Maximum Likelihood Estimation, MLE）方法来进行强度参数估计：

$$L(\lambda) = \prod_{(i,j)\in \text{pairs}}
\left( \frac{\lambda_i}{\lambda_i + \lambda_j} \right)^{x_{ij}}
\left( \frac{\lambda_j}{\lambda_i + \lambda_j} \right)^{1 - x_{ij}}$$

其中 $x_{ij}$ 是指示变量，如果 $O_i$ 对 $O_j$ 获胜，则 $x_{ij} = 1$，否则 $x_{ij} = 0$。

（至此，必要的预备知识我们已经具备，接下来就开始推导~

## 3.DPO的推导过程
### 3.1 基于 BT 的损失函数构建
设 $y_w$ 是人类偏好中的优选，$y_l$ 是次优选。那么，根据 BT 模型，人类偏好 $y_w$ 优于 $y_l$ 的概率为：

$$P(y_w \succ y_l \mid x) = \dfrac{\lambda_w}{\lambda_w + \lambda_l}$$

但是在强化学习中，我们用什么来表示 $\lambda$ 呢？聪明的你一定想到：我们可以用奖励模型的分数 $r(x, y)$。考虑到 $r(x, y)$ 可能为负，所以用指数函数表示强度 $e^{r(x,y)}$，那么上述式子可以写成：

$$P(y_w \succ y_l \mid x) =
\dfrac{e^{r(x,y_w)}}{e^{r(x,y_w)} + e^{r(x,y_l)}}$$

上式分子分母同时除以 $e^{r(x,y_w)}$，得到：

$$P(y_w \succ y_l \mid x)
= \dfrac{1}{1 + e^{r(x,y_l) - r(x,y_w)}}
= \sigma\!\left(r(x,y_w) - r(x,y_l)\right)$$

其中 $\sigma$ 是 sigmoid 函数：

$$\sigma(x) = \dfrac{1}{1 + e^{-x}}$$

于是对于所有的数据对 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^{N}$，我们可以进行极大似然估计：

$$\mathcal{L}(r, \mathcal{D})
= \prod_{(x,y_w,y_l)\in\mathcal{D}}
\sigma\!\left(r(x,y_w) - r(x,y_l)\right)$$

然后对等式两边同时取对数，并取平均，原先的极大似然估计就转换成了负对数似然损失（negative log-likelihood loss）：

$$\mathcal{L}_{\mathrm{CR}}(r_\phi, \mathcal{D})
= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\log \sigma\!\left(r_(x,y_w) - r_(x,y_l)\right)$$

(请注意：尽管在理想化假设下，通过极大似然估计可以推导出最优策略在分布空间中的闭式解，但由于实际训练中策略被限制在参数化函数族内，我们仍需将该目标转化为损失函数，通过梯度优化在参数空间中近似该最优解。

**现在我们成功地将矛盾转换到这个奖励函数上，只要成功得到奖励函数，一切也就迎刃而解了！而奖励函数的获取也正是DPO的灵魂所在~**

### 3.2 奖励函数的推导
从理论上看，DPO 将奖励函数从一个需要估计的对象转化为策略优化过程中的隐变量，这是其区别于其他偏好优化方法的关键所在。下面我们将一层层揭开他的面纱：

#### 3.2.1 先从一个常见的RL目标开始：
在很多RLHF的表述中，优化目标为：
$$
\max_{\pi_\theta}
\;\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y \mid x)}
\big[
r_\phi(x, y)
\big]
-
\beta \, D_{\mathrm{KL}}\!\left(
\pi_\theta(y \mid x)\,\|\,\pi_{\mathrm{ref}}(y \mid x)
\right)
$$ 
其中：

- $\pi_\theta(y \mid x)$ 是当前策略模型的输出分布；
- $\pi_{\mathrm{ref}}(y \mid x)$ 是参考策略（通常是初始模型）；
- $r_\phi(x, y)$ 是隐式奖励模型。

上述优化目标的根本目的为想要寻找一个最佳的策略 $\pi_\theta$ 使得最终获得的奖励期望高，同时又不太过于远离 $\pi_{ref}$ （防止训偏）

#### 3.2.2 详细推导
$$
\begin{aligned}
\max_{\pi}\;&
\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(y \mid x)}
\big[
r(x,y)
\big]
-
\beta D_{\mathrm{KL}}\!\left(
\pi(y \mid x)\,\|\,\pi_{\mathrm{ref}}(y \mid x)
\right) \\[6pt]

= \max_{\pi}\;&
\mathbb{E}_{x \sim \mathcal{D}} \,
\mathbb{E}_{y \sim \pi(y \mid x)}
\Big[
r(x,y)
-
\beta \sum_{x \sim \mathcal{D},\, y \sim \pi(y \mid x)}
\pi(y \mid x)
\log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
\Big] \\[6pt]

= \max_{\pi}\;&
\mathbb{E}_{x \sim \mathcal{D}} \,
\mathbb{E}_{y \sim \pi(y \mid x)}
\Big[
r(x,y)
-
\beta \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
\Big] \\[6pt]

= \min_{\pi}\;&
\mathbb{E}_{x \sim \mathcal{D}} \,
\mathbb{E}_{y \sim \pi(y \mid x)}
\Big[
\log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
-
\frac{1}{\beta} r(x,y)
\Big] \\[6pt]

= \min_{\pi}\;&
\mathbb{E}_{x \sim \mathcal{D}} \,
\mathbb{E}_{y \sim \pi(y \mid x)}
\Big[
\log
\frac{
\pi(y \mid x)
}{
\frac{1}{Z(x)} \,
\pi_{\mathrm{ref}}(y \mid x)
e^{\frac{1}{\beta} r(x,y)}
}
-
\log Z(x)
\Big]
\end{aligned}
$$

其中：

$Z(x) = \sum_{y} \pi_{\mathrm{ref}}(y \mid x)\,
e^{\frac{1}{\beta} r(x,y)}$

观察到式子中的分母，我们可以定义：

$$\pi^{*}(y \mid x)
= \dfrac{1}{Z(x)}\,
\pi_{\mathrm{ref}}(y \mid x)\,
e^{\frac{1}{\beta} r(x,y)}$$

注意到 $Z(x)$ 和 $\pi$ 无关，则原式可写为：

$$
\begin{aligned}
\min_{\pi}\;&
\mathbb{E}_{x \sim \mathcal{D}}
\mathbb{E}_{y \sim \pi(y \mid x)}
\Big[
\log
\frac{
\pi(y \mid x)
}{
\frac{1}{Z(x)}\,
\pi_{\mathrm{ref}}(y \mid x)\,
e^{\frac{1}{\beta} r(x,y)}
}
-
\log Z(x)
\Big] \\[6pt]

= \min_{\pi}\;&
\mathbb{E}_{x \sim \mathcal{D}}
\mathbb{E}_{y \sim \pi(y \mid x)}
\Big[
\log
\frac{
\pi(y \mid x)
}{
\pi^{*}(y \mid x)
}
-
\log Z(x)
\Big] \\[6pt]

= \min_{\pi}\;&
\mathbb{E}_{x \sim \mathcal{D}}
\Big[
D_{\mathrm{KL}}\!\left(
\pi(y \mid x)\,\|\,\pi^{*}(y \mid x)
\right)
-
\log Z(x)
\Big]
\end{aligned}
$$

因为 $Z(x)$ 与 $\pi$ 无关，对于 KL 散度，当且仅当
$\pi(y \mid x) = \pi^{*}(y \mid x)$ 时取最小值。

所以可知在我们 RL 目标的基础上，可得到的最优策略即为：

$$\pi(y \mid x) = \pi^{*}(y \mid x)
= \dfrac{1}{Z(x)} \,
\pi_{\mathrm{ref}}(y \mid x)\,
e^{\frac{1}{\beta} r(x,y)}$$

变形一下上述式子，我们有：
$$r(x,y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
+ \beta \log Z(x)$$

因为 $Z(x)$ 和 $\pi$ 无关，也就是说**它只跟 prompt 有关，对比较同一个 $x$ 下的两个回答时会抵消。**

#### 3.2.3 奖励函数带回3.1中所提的损失函数
在3.1中我们已经知道，最后的损失函数为：
$$\mathcal{L}_{\mathrm{CR}}(r_\phi, \mathcal{D})
= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\log \sigma\!\left(r_(x,y_w) - r_(x,y_l)\right)$$

因此，将我们上述推导得到的奖励函数带回即可得到：
$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log \sigma\!\left(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
-
\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right)
\right]
$$

上面的式子是不是看起来丑丑的，换种形式来看下呢：
$$
\mathcal{L}_{\mathrm{DPO}}(\theta)
= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log \sigma\!\left(
\beta \left(
\log \pi_\theta(y_w \mid x)
-
\log \pi_\theta(y_l \mid x)
\right)
-
\left(
\log \pi_{\mathrm{ref}}(y_w \mid x)
-
\log \pi_{\mathrm{ref}}(y_l \mid x)
\right)
\right)
\right]
$$

其中：

- $\pi_\theta(y \mid x)$ 是当前策略；
- $\pi_{\mathrm{ref}}(y \mid x)$ 是参考策略。


直觉上：**DPO 优化的不是更偏向优选回答，而是相对于参考策略更有把握地偏向优选回答**。

## DPO 数据集格式参考
```json
[
dpo_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Java",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "C++",
    ],
}
]
```

## DPO训练示例代码
```python
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# -----------------------------
# 1. 构造 DPO 偏好数据
# 每条数据包含：
#   - prompt
#   - chosen    (人类偏好回答 y_w)
#   - rejected  (人类不偏好回答 y_l)
# -----------------------------
train_dataset = Dataset.from_dict({
    "prompt": prompt_list,
    "chosen": chosen_list,
    "rejected": rejected_list,
})

# -----------------------------
# 2. DPO 训练配置
# beta 对应理论中的 KL 正则强度
# -----------------------------
training_args = DPOConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    beta=0.1,
    logging_steps=1,
    output_dir="./dpo_ckpt",
)

# -----------------------------
# 3. （可选）参数高效微调配置
# -----------------------------
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

# -----------------------------
# 4. 构造 DPO Trainer 并训练
# -----------------------------
dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

dpo_trainer.train()
```

## 参考资料：
https://zhuanlan.zhihu.com/p/779691018
https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN





