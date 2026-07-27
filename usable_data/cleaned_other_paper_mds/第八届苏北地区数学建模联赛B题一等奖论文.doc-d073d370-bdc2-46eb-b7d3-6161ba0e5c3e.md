本文围绕五一黄金周的旅游问题进行了定量评估，对无时限的旅游费用问题、无费用限制的旅游时间问题、有费用限制的旅游质量问题、有时限的旅游质量问题、既有时限又有费用限制的旅游质量问题分别建立了数学模型并设计了旅游行程表，对求解结果进行了分析。

问题一放开了对时间的限制，要求设计一条用尽可能少的费用游览十个景点的旅游线路。首先，我们对预选的旅游景点之间消耗的费用和时间进行了分析。由于约束条件只要求费用最低，因此我们从火车和长途汽车班次中选取费用最低的并记录下来建立了最优通行费表。第二步，根据Hamilton回路算法的有关方法，以费用为参考量，我们建立了一个适用于本问题最优规划模型。第三步，用C语言编写模型的指令，运行后得到最优旅游路线：  $\rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow$  ；第四步，综合考虑安排，建立行程表；计算可得最少的总旅行费用为3101元。

问题二在不限制费用的条件下，要求用最短的时间游览完十个景点。其原理与问题一非常相似，故可用问题一的数学模型及方法，改用景点之间消耗的时间作为参考量，最终得到行程表且知最优旅游路线：  $\rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow$  ；最短的旅行总时间  $T = 8$  天22小时23分。

问题三要求我们在只有2000元旅游费用的条件下游览尽可能多的城市。因此我们引入0一1变量表示是否游览某个景点，从而推出交通费用和景点花费的函数表达式，给出相应的约束条件。这样寻找不同景点数时的最优旅游路线，并计算其总费用。则最优旅游路线的总花费为1795元，游览了7个景点，是不超过2000元的最大值，据此构建行程表。

问题四中我们要在5天的时间内游览最多的景点并回到徐州。其实质是把问题三中

的费用约束条件变成了时间约束，故在此我们依然可用问题三中的模型进行求解，得到最多可游览6个景点，耗时4天13小时（106小时），据此建立行程表。

问题五可看做是问题三、四的合并，其中费用和时间都是约束条件。因此我们综合问题三、四中的算法，运用问题三中的模型对其进行全面分析，得到最多可游览6个景点，并建立行程表。

##############

1. 问题重述随着人们的生活不断提高，旅游已成为提高人们生活质量的重要活动。江苏徐州有一位旅游爱好者打算现在的今年的五月一日早上8点之后出发，到全国一些著名景点旅游，最后回到徐州。由于跟团旅游会受到若干限制，他(她)打算自己作为背包客出游。他预选了十个省市旅游景点，如表1所示。

表1.预选的十个省市旅游景点  


# 假设：

(A)城际交通出行可以乘火车(含高铁)、长途汽车或飞机（不允许包车或包机），并且车票或机票可预订到。

(B)市内交通出行可乘公交车(含专线大巴、小巴)、地铁或出租车。

(C) 旅游费用以网上公布为准，具体包括交通费、住宿费、景点门票(第一门票)。晚上20:00至次日早晨7:00之间，如果在某地停留超过6小时，必须住宿，住宿费用不超过200元/天。吃饭等其它费用60元/天。

(D) 假设景点的开放时间为8:00至18:00。

问题：

根据以上要求，针对如下的几种情况，为该旅游爱好者设计详细的行程表，该行程表应包括具体的交通信息(车次、航班号、起止时间、票价等)、宾馆地点和名称，门票费用，在景点的停留时间等信息。

(1) 如果时间不限，游客将十个景点全游览完，至少需要多少旅游费用？请建立相关数学模型并设计旅游行程表。

(2) 如果旅游费用不限，游客将十个景点全游览完，至少需要多少时间？请建立相关数学模型并设计旅游行程表。

(3) 如果这位游客准备2000元旅游费用，想尽可能多游览景点，请建立相关数学模型并设计旅游行程表。

(4) 如果这位游客只有5天的时间，想尽可能多游览景点，请建立相关数学模型并设计旅游行程表。

(5) 如果这位游客只有5天的时间和2000元的旅游费用，想尽可能多游览景点，请建立相关数学模型并设计旅游行程表。

# 2.模型的假设与符号说明

# 2.1 模型的假设

五一黄金周正值旅游旺季，各地旅游景点吸引了大批游客前往观光。考虑到该游客的旅游路线跨越区域较大，交通情况尚存在一些不确定因素。为了研究方便，我们给出以下假设：

(1) 城际交通出行可以乘火车(含高铁)、长途汽车或飞机(不允许包车或包机)，并且车票或机票可预订到；

(2) 市内交通出行可乘公交车(含专线大巴、小巴)、地铁或出租车；

(3) 旅游费用以网上公布为准，具体包括交通费、住宿费、景点门票(第一门票)，晚上20：

00至次日早晨7：00之间，如果在某地停留超过6小时，必须住宿，住宿费用不超过200元/天。吃饭等其它费用60元/天；

(4) 假设景点的开放时间为8:00至18:00；

(5) 假设火车、汽车和飞机均正点到达，行程中无事故、无阻碍；

(6) 假设由火车换乘汽车或者汽车换乘火车的时间很短，忽略不计；

(7) 假设旅游过程中天气条件良好，不影响行程；

(8) 由于考虑到在城市内有时需坐公交（大巴）有时需坐出租车，经过近似计算，取每个城市内交通费用为10元。

2.2 模型的符号说明

(1) i,j表示第i个城市（景点）或第j个城市（景点），i，j=0,1,2……10；

(2) Z表示计划行程中的总费用；

(3) W表示各城市（景点）之间的交通费用的总和， $W_{ij}$ 表示各城市（景点）之间的交通费用；

(4) A表示在景点所在城市的总花费，其中包括 $M_{i}$ 表示第i个城市（景点）内的交通费用， $S_{i}$ 表示第i个城市（景点）内的食宿费用， $G_{i}$ 表示第i个城市（景点）内的景点门票费用， $A_{i}$ 表示第i个城市（景点）内的总费用，故 $A_{i} = M_{i} + S_{i} + G_{i}$ ；

(5) t表示在第i个城市（景点）的逗留时间，tij表示从第i个景点到第j个景点路途中所需时间，T表示本次旅游的总时间；

(6)  $\mathbf{r}_{ij} = \left\{ \begin{array}{ll}1 & \text{游客直接从第i个到第j个景点} \\ 0 & \text{其他} \end{array} \right.$

# 3. 问题的分析

# 3.1 问题背景的分析

根据对题目的理解我们知道，旅游时的总费用包括交通费用、住宿费用和在景点旅游时的费用，在研究确定旅游路线和选用的交通工具后，我们的目标就是在所有的约束条件情况下，求出所求目标的最优解。

3.2 对问题一和问题二的分析

问题一要求我们在不限定时间的情况下，游览完十个景点，并设计出花费最少的旅游路线，故要尽量选择便宜的交通工具。这里我们的做法是以任意两景点间的交通费用为权值，构建一个完备图；然后利用Hamilton回路算法计算出近似最佳旅游路线，进而得出最佳方案。

问题二实质上是在问题一的基础上改变了约束条件，在不限资金的条件下尽快结束十个景点的旅程。故可用与问题一类似的方法，且应尽量乘坐飞机以减少时间。

# 3.3对问题三和问题四的分析

经过分析，我们可以知道这两个问题所要实现的目标是，使游客在规定的时间内和规定的花费内游览尽可能多的地方。游览的总费用由两部分组成，分别为交通总费用和在旅游景点的花费。

对于问题三，花费在2000元以内且游览的景点尽量多是该问题的目标。因此，我们的做法是在满足相应的约束条件下，先确定游览的景点数，然后利用Hamilton回路算法和0- - 1模型计算出在这种情况下的最小花费，这样最终会得出几种旅游路线。问题四中，花费在2000元以内的条件改为限定时间为最多5天，故可使用与问题三类似的方法求得最优解。

# 3.4对于问题五的分析

问题五是对问题三和问题四进一步综合，要求我们用5天的时间和2000元的旅游费用游览尽可能多的景点。故可采用与问题三、四类似的方法，进行综合性的求解。

# 4.模型的准备

先给11个旅游城市分别进行编号，徐州、常州、青岛、北京、祁县、洛阳、黄山、武汉西安、九江、舟山分别编为、、、、、、，则这11个城市和其交通线路构成了一个网络图。这些城市可看作该网络图的节点，这些节点由相应的交通线路相连，节点之间的边就是交通线路。


4.20——1模型

4.2.1目标函数的确立：

游览的总费用由2部分组成，分别为交通总费用和在旅游景点的花费。我们已经定义Z——旅游总花费；W——交通总费用；A——旅游景点的花费；

从而得到目标函数：

$$
\mathrm{Z} = \mathrm{W} + \mathrm{A}
$$

# （1）交通总花费

因为  $\mathrm{W}_{ij}$  表示第i个景点到第j个景点所需的交通费用，而  $\mathrm{t}_{\mathrm{i}}$  是判断游客们是否从第i个景点直接到第j个景点的0——1变量，因此我们可以很容易的得到交通总费用为：

$$
\mathrm{W} = \sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times \mathrm{W}_{\mathrm{ij}}
$$

(2）旅游景点的花费

因为  $\mathrm{A}_{\mathrm{i}}$  表示游客在i个景点的总消费，  $\mathbf{r}_{\mathrm{ij}}$  也可以表示出是否到达过第i个和第j个景点，而整个旅游路线又是一个环形，因此  $\sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times (\mathrm{A}_{\mathrm{i}} + \mathrm{A}_{\mathrm{j}})$  实际上将所到景点的花费计算了两遍，从而我们可以得到旅游景点的花费为：

$$
\mathrm{A} = \frac{1}{2}\times \sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times (\mathrm{A}_{\mathrm{i}} + \mathrm{A}_{\mathrm{j}})
$$

从而我们可以得到目标函数为：

$$
\mathrm{Min}\qquad \mathrm{Z} = \mathrm{W} + \mathrm{A}
$$

4.2.2约束条件：

$①$  时间约束

旅游时间应该不超过5天，而这些时间包括在路途中的时间和在旅游景点逗留的时间。因为  $\mathbf{t}_{\mathrm{ij}}$  表示从第i个景点到第j个景点路途中所需时间，所以路途中所需的总时间为 $\sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times \mathrm{t}_{\mathrm{ij}}$  ；  $\mathbf{t}_{\mathrm{i}}$  表示在第i个景点的逗留时间，故在旅游景点的总逗留时间为 $\frac{1}{2}\times \sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times (\mathrm{t}_{\mathrm{i}} + \mathrm{t})$  。因此，总的时间约束为：

$$
\sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times \mathrm{t}_{\mathrm{ij}} + \frac{1}{2}\times \sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times (\mathrm{t}_{\mathrm{i}} + \mathrm{t})\leq 120
$$

$②$  旅游景点数约束

根据假设，整个旅游路线是环形，即最终要回到徐州，因此  $\sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}$  即表示旅游的景点数，这里我们假定要旅游的景点数为n  $(n = 1,2,3,\dots ,10)$  。因此旅游景点数约束为：

$$
\sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}} = \mathrm{n}\qquad (\mathrm{n} = 1,2,3\cdot \dots \cdot 10)
$$

$③0$  ——1变量约束[3]

我们可以把所有的景点连成一个圈，而把妹一个景点看做圈上一个点。对于每个景点来说，只允许最多一条边进入，同样只允许最多一条边出来，并且有一条边进入就要

有一条边出去。因此可得约束：

$$
\sum_{\mathrm{i}}\mathrm{r_{ij}} = \sum_{\mathrm{j}}\mathrm{r_{ij}}\leq 1\qquad (\mathrm{i},\mathrm{j} = 0,1,2,\dots ,10)
$$

当  $\mathrm{i} = 1$  时，因为徐州是出发点，所以  $\sum_{\mathrm{i = 0}}^{\mathrm{r_{ij}} = 1};\mathrm{j = 1}$  时，因为最终要回到徐州，所以  $\sum_{\mathrm{j = 0}}^{\mathrm{r_{ij}} = 1}$  。综上所述，我们可以得到总的模型为：

Min  $\mathrm{Z} = \mathrm{W} + \mathrm{A}$

$$
= \sum_{\mathrm{i = 0}}^{\mathrm{10}}\sum_{\mathrm{j = 0}}^{\mathrm{10}}\mathrm{r_{ij}}\times \mathrm{W_{ij}} + \frac{1}{2}\times \sum_{\mathrm{i = 0}}^{\mathrm{10}}\sum_{\mathrm{j = 0}}^{\mathrm{10}}\mathrm{r_{ij}}\times (\mathrm{A_{i}} + \mathrm{A_{j}})
$$

约束条件：

$$
\begin{array}{r l} & {\left\{ \begin{array}{l l}{\sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times \mathrm{t}_{\mathrm{ij}} + \frac{1}{2}\times \sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times (\mathrm{t}_{\mathrm{i}} + \mathrm{t}_{\mathrm{j}})\leq 120}\\ {\sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}} = \mathrm{n}\quad \quad (\mathrm{n} = 1,2\dots 10)}\\ {\sum_{\mathrm{i} = 0}^{10}\mathrm{r}_{\mathrm{ij}} + \sum_{\mathrm{j}}\mathrm{r}_{\mathrm{ij}}\leq 1\quad \quad (\mathrm{i},\mathrm{j} = 0,1\dots 10)}\\ {\sum_{\mathrm{i} = 0}^{1}\mathrm{r}_{\mathrm{ij}} = 1}\\ {\sum_{\mathrm{i} = 0}^{1}\mathrm{r}_{\mathrm{ij}} = 1}\\ {\sum_{\mathrm{j} = 0}^{1}\mathrm{r}_{\mathrm{ij}} = 1}\\ {\sum_{\mathrm{i} = 0}^{1}\mathrm{r}_{\mathrm{ij}} = 1}\\ {\sum_{\mathrm{j} = 0}^{1}\mathrm{r}_{\mathrm{ij}} = 1} \end{array} \right.} \end{array}
$$

# 各大景点门票信息[4]


# 5.模型的建立与求解

5.1建立无时限的旅游费用Hamilton回路模型

根据问题一中的约束条件，由于要求在没有时间限制的条件下旅行，因此为了保证游完十个景点所花费用最少，我们选择了耗资最少的方式旅行：首先在选择交通工具时飞机的费用明显过高，予以排除，从现有火车和汽车方案中选择便宜的进行计算；其次，在景点所在城市尽量减少住宿费和餐饮费。根据此思路，搜集资料得出任意两景点之间

费(元)</td><td>徐州</td><td>常州</td><td>青岛</td><td>北京</td><td>祁县</td><td>洛阳</td><td>黄山</td><td>武汉</td><td>西安</td><td>九江</td><td>舟山
(宁波)</td><td>停留
时间</td></tr><tr><td>徐州</td><td>0</td><td>34</td><td>70</td><td>53</td><td>/</td><td>34</td><td>99</td><td>/</td><td>55</td><td>50</td><td>130</td><td>0小时</td></tr><tr><td>常州</td><td>34</td><td>0</td><td>150</td><td>78</td><td>/</td><td>125</td><td>73</td><td>199</td><td>165</td><td>173</td><td>73</td><td>4小时</td></tr><tr><td>青岛</td><td>70</td><td>150</td><td>0</td><td>116</td><td>/</td><td>125</td><td>182</td><td>/</td><td>165</td><td>170</td><td>350</td><td>6小时</td></tr><tr><td>北京</td><td>53</td><td>78</td><td>116</td><td>0</td><td>53</td><td>53</td><td>182</td><td>280</td><td>136</td><td>145</td><td>332</td><td>3小时</td></tr><tr><td>祁县</td><td>/</td><td>/</td><td>/</td><td>53</td><td>0</td><td>/</td><td>/</td><td>/</td><td>41</td><td>/</td><td>/</td><td>3小时</td></tr><tr><td>洛阳</td><td>34</td><td>125</td><td>125</td><td>53</td><td>/</td><td>0</td><td>/</td><td>87</td><td>28</td><td>62</td><td>/</td><td>3小时</td></tr><tr><td>黄山</td><td>99</td><td>73</td><td>182</td><td>182</td><td>/</td><td>/</td><td>0</td><td>78</td><td>/</td><td>68</td><td>164</td><td>7小时</td></tr><tr><td>武汉</td><td>/</td><td>199</td><td>/</td><td>280</td><td>/</td><td>87</td><td>78</td><td>0</td><td>137</td><td>51</td><td>300</td><td>2小时</td></tr><tr><td>西安</td><td>55</td><td>165</td><td>165</td><td>136</td><td>41</td><td>28</td><td>/</td><td>137</td><td>0</td><td>70</td><td>194</td><td>2小时</td></tr><tr><td>九江</td><td>50</td><td>173</td><td>170</td><td>145</td><td>/</td><td>62</td><td>68</td><td>51</td><td>70</td><td>0</td><td>115.5</td><td>7小时</td></tr><tr><td>舟山
(宁波)</td><td>130</td><td>73</td><td>350</td><td>332</td><td>/</td><td>/</td><td>164</td><td>300</td><td>194</td><td>115</td><td>0</td><td>6小时</td></tr></table>

的最优通行费用表（见下表），以表内费用值作为Hamilton回路图中各边的权值。

注：“/”代表耗费时间、金钱明显过多的路线，不考虑在内

编写基于Hamilton回路算法的C语言程序，输入上表数据（“/”一律按500输入）运行得出无限时条件下的最优路线方案如下图：


故旅游的最优城市顺序为：→→→→→→→→→→→

进一步规划，综合考虑，得出行程表：


城市之间的交通费

$$
W = W_{01} + W_{0,10} + W_{10,9} + W_{96} + W_{67} + W_{75} + W_{58} + W_{84} + W_{43} + W_{32} + W_{20}
$$

$$
= 34 + (73 + 32 + 2) + 94 + (76 + 130) + (49 + 111) + 87 + 28 + 39 + 53 + 116 + 99
$$

$= 1053$  （元）

市内的交通费

$$
M = M_{1} + M_{2} + M_{3} + M_{4} + M_{5} + M_{6} + M_{7} + M_{8} + M_{9} + M_{10}
$$

食宿费

$$
S = S_{1} + S_{2} + S_{3} + S_{4} + S_{5} + S_{6} + S_{7} + S_{8} + S_{9} + S_{10} = 60 \times 11 + 80 = 740 \text{（元）}
$$

景点门票费

$$
G = G_{1} + G_{2} + G_{3} + G_{4} + G_{5} + G_{6} + G_{7} + G_{8} + G_{9} + G_{10} = 1210
$$

所以总旅行费用

$$
Z = W + M + S + G = 1053 + 100 + 740 + 1210 = 3101
$$

5.2 建立无费用限制的旅游时间 Hamilton 回路模型

问题二要求我们不限旅游费用，用最短的时间游完十个景点并回到徐州。分析了此问题的约束条件，我们应建立模型近似得出耗时最少的方案。注意到：飞机、动车和高铁相比其他运输方式耗时少，所以优先考虑；此外还要尽最大可能规划以减少在景点的时间以及住宿时间。综合多方面因素，通过各种渠道，我们建立了任意两景点之间的最优耗时表（如下表）：

(宁波)</td><td>停留
时间</td></tr><tr><td>徐州</td><td>0</td><td>3.36</td><td>2</td><td>1.5</td><td>24</td><td>2.66</td><td>2.08</td><td>2.41</td><td>2.41</td><td>2.08</td><td>1.91</td><td>0小时</td></tr><tr><td>常州</td><td>3.36</td><td>0</td><td>2.91</td><td>1.75</td><td>24</td><td>3.41</td><td>2.66</td><td>3.16</td><td>1.83</td><td>2.83</td><td>2.75</td><td>4小时</td></tr><tr><td>青岛</td><td>2</td><td>2.91</td><td>0</td><td>1.33</td><td>24</td><td>3.16</td><td>1.83</td><td>2.08</td><td>1.91</td><td>2.5</td><td>2.33</td><td>6小时</td></tr><tr><td>北京</td><td>1.5</td><td>1.75</td><td>1.33</td><td>0</td><td>13.7</td><td>1.75</td><td>2</td><td>2.08</td><td>2</td><td>2.33</td><td>1.91</td><td>3小时</td></tr><tr><td>祁县</td><td>24</td><td>24</td><td>24</td><td>13.7</td><td>0</td><td>24</td><td>24</td><td>24</td><td>10.9</td><td>24</td><td>24</td><td>3小时</td></tr><tr><td>洛阳</td><td>24</td><td>3.41</td><td>3.17</td><td>1.75</td><td>24</td><td>0</td><td>3.08</td><td>2.83</td><td>2.67</td><td>3.08</td><td>2.67</td><td>3小时</td></tr><tr><td>黄山</td><td>2.08</td><td>2.66</td><td>1.83</td><td>2</td><td>24</td><td>3.08</td><td>0</td><td>2.42</td><td>2.17</td><td>2.08</td><td>1.92</td><td>7小时</td></tr><tr><td>武汉</td><td>2.42</td><td>3.16</td><td>2.08</td><td>2.08</td><td>24</td><td>2.83</td><td>2.42</td><td>0</td><td>1.42</td><td>0.58</td><td>2.33</td><td>2小时</td></tr><tr><td>西安</td><td>2.42</td><td>1.83</td><td>1.92</td><td>2</td><td>10.9</td><td>2.67</td><td>2.17</td><td>1.42</td><td>0</td><td>3.08</td><td>2.92</td><td>2小时</td></tr><tr><td>九江</td><td>2.08</td><td>2.83</td><td>2.5</td><td>2.33</td><td>24</td><td>3.08</td><td>2.08</td><td>0.58</td><td>3.08</td><td>0</td><td>1.92</td><td>7小时</td></tr><tr><td>舟山
(宁波)</td><td>1.92</td><td>2.75</td><td>2.33</td><td>1.92</td><td>24</td><td>2.67</td><td>1.92</td><td>2.33</td><td>2.92</td><td>1.92</td><td>0</td><td>6小时</td></tr></table>

注：表中注为“24"小时的格子代表耗时明显较多的路线，不予考虑

编写基于 Hamilton 回路算法的 C 语言程序，输入上表数据，运行得出无费用限制条件下的最优路线方案如下图：


故旅游的最优城市顺序为：  $\rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow$  进一步规划，综合考虑，得出行程表：



所以总旅行时间为从徐州出发至回到徐州的时间，即为 $\mathrm{T} = 8$  天22小时23分

5.3建立有费用限制的0——1模型旅游质量Hamilton回路模型由于问题三只限制总费用，对时间未作限制，故0——1模型没有限制条件：

$$
\sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times \mathrm{t}_{\mathrm{ij}} + \frac{1}{2}\times \sum_{\mathrm{i} = 0}^{10}\sum_{\mathrm{j} = 0}^{10}\mathrm{r}_{\mathrm{ij}}\times (\mathrm{t}_{\mathrm{i}} + \mathrm{t}_{\mathrm{j}})\leq 120
$$

结合旅游质量Hamilton回路模型计算可得下表：


由上表可知在限制2000元花费时，最多浏览7个景点。

根据已知数据安排行程表  



# 5.4 建立有时限的旅游质量 Hamilton 回路模型

由于只限制时间而没要求花费，故只将目标函数作为参照，通过结合 Hamilton 回路模型计算可得下表：

数n</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>时间（小
时）</td><td>84</td><td>99</td><td>109</td><td>128</td></tr><tr><td>路线</td><td>0→4→8→
5→3→0</td><td>0→3→4→
8→7→5→
0</td><td>0→8→3→
4→5→7→
9→0</td><td>0→2→8→3→4→5→7→
9→0</td></tr></table>

由上表可知在限制5天（120小时）时，最多浏览6个景点。

根据所得数据建立行程表：



5.5 建立既有时限又有费限的旅游质量 Hamilton 回路模型由于规定了费用 2000 元和时间 5 天，所以这是一个完整的 0——1 模型，通过对综合模型和结合 Hamilton 回路模型的运算，可得下表：

数n</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>总花费
(元)</td><td>876</td><td>1132</td><td>1988</td><td>2238</td></tr><tr><td>路线</td><td>0→8→4→
5→3→0</td><td>0→8→4→
7→3→5→
0</td><td>0→3→8→
4→5→7→
9→0</td><td>0→2→8→5→4→3→9→
7→0</td></tr></table>

根据已知数据，设计行程表：



# 6.模型的结果分析

问题一：推荐最优旅游路线： \(\rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow\) 旅游总费用：3101元问题二：推荐最优旅游路线： \(\rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow\) 旅游总耗时：8天22小时23分问题三：推荐最优旅游路线： \(\rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow\) 旅游景点数：7 旅游总费用：1795元问题四：推荐最优旅游路线： \(\rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow\)\) 旅游景点数：6 旅游总耗时：4天13小时（109小时）问题五：推荐最优旅游路线： \(\rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow \rightarrow_{\rightarrow}\) 旅游景点数：6 旅游总费用：1988元旅游总耗时：4天22小时（118小时）

本文通过建立基于Hamilton回路算法的旅游路线模型和引入0一1模型进行规划，在五种不同的约束条件下为游客设计了不同的近似最优旅行路线。由于用了0一1模型进行简化，建模和编程得以顺利完成；经后期检验，所得结果能满足题目的要求，最大程度减少了时间或资金的消耗，具有较好的实际意义。但由于数据量过于庞大，模型中为了方便研究又有一些假设，所以所得结果只是近似最优解。

# 7.模型的评价

7. 模型的评价本文根据游客的旅行路线进行了合理假设，简化了次要因素，把问题转化为图论上最佳旅行商回路问题解决，思路比较清晰，模型恰当，得出的方案相对合理，使问题得到

了比较合理的解决；成功的使用了0——1变量，使模型的建立和求解得以顺利进行。但是，由于数据庞大，对程序的要求很高，尽管经过了检验，但结果依然比较粗糙，有待进一步的改进。实际情况中，两景点之间的交通方式比较复杂，如公路、铁路、航班之间可以转换，增加这些考虑后，结果会更加合理。且数据资料搜集的不完整，有一定的局限性准确性也有待商榷，而且没有对最终方案进行更为细致的研究讨论，这些方面还有待改进。
