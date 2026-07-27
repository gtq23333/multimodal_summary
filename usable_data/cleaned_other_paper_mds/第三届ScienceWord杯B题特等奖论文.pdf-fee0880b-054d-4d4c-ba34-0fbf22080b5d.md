本文用随机线性及非线性动态规划方法研究北京二环高峰时段交通堵塞的原因，并用自适应系统的模拟仿真实验验证理论模型。主要结果表明：1. 二环路的道路拥堵符合“Braess悖论”描述的情况，即当高峰时段二环以内虽然许多路段（东西走向）开通以缓解市区交通压力，但这种做法却降低了整个二环线的交通运行效率，造成多个主要南北走向干道（如文慧桥至复兴门桥、建国门至左安门等路段）交通流的不稳定及车辆拥堵，实际交通效率下降。2. 交通堵塞问题并不会因司机广泛使用GPS导航系统而得到缓解，因为司机在缺乏或过分依赖GPS导航选择驾驶路线都可能导致“Braess悖论”问题，司机接受GPS导航建议的程度会影响交通网络的运行效率。3. 这一结论在细胞自适应系统的仿真模拟实验中得到验证。GPS导航系统使司机能及时掌握当前路况和最新行车路线，并在此基础上结合自身经验作出判断，虽然提高了临时路段的局部通行能力，却使所有出行者的出行时间都增加了。当GPS导航和司机自身的判断达到一定比例时约  $10\%$  ，全部车辆的行驶时间最接近社会最优的通过时间。所以，“Braess”悖论是否发生不仅和个体是否了解全局状况有关，而且还与司机是否遵循GPS建议的最优路线行驶密切相关。

参赛队号

所选题目

![](images/df162794c5213ffb8d5f8ef6cb15bfb2e594960c72c4c386211ab016e7a1323a.jpg)

In this paper the causes of traffic jams during peak hours in the second ring of Beijing are analyzed by stochastic dynamic programming (linear and non- linear) and the simulation of self- adaptive multi- agents is carried out to verify our findings. The three conclusions are as follows. First, "Braess Paradox" is found in the traffic blockings of the second ring. The east- to- west roads that are designed to relieve traffic congestion actually cause the instability of car flows on some south- to- north roads (e.g. from Wenhui Bridge to Fuxingmen Bridge, from Jianguomen to Zuoanmen), thus bringing more traffic jams and making the public transportation inefficient. Second, whether the GPS navigation system is helpful in solving the traffic jams depends on the level of acceptance for the drivers. "Braess Paradox" is still possible due to the externality caused by all the drivers. Last, the conclusion is confirmed in the simulation data of a self- adaptive system. If the drivers choose their optimal paths based on both the GPS navigation recommendations and their own experience, the self- adaptive optimization behavior will still suffer from a social welfare loss. However, when the suggestion given by GPS navigation system is denied by  $10\%$  of the drivers, the time all the cars take to go though the paths is the closet to the social optimal time spending. In summary, both the traffic situation provided for the drivers and their acceptance of those paths are related with "Braess Paradox" in public transportation.

##############

1 问题背景Dietrich Braess 在 1968 年的一篇文章中提出了道路交通体系当中的 Braess 悖论。它的含义是：有时在一个交通网络上增加一条路段，或者提高某个路段的局部通行能力，反而使所有出行者的出行时间都增加了，这种为了改善通行能力的投入不但没有减少交通延误，反而降低了整个交通网络的服务水平。人们对这个问题做过许多研究，在成都市建设当中也尽量避免这种现象的发生。但是在复杂的城市道路当中，Braess 导论仍然不时出现，造成实际交通效率的显著下降。在此，请你通过合理的模型来研究和解决城市交通中的 Braess 悖论。

# 2 问题提出：

（1)通过健实城市的道路交通情况，建立合理的模型，判断在北京市二环路以内的路网中（包括二环路）出现的交通拥堵，是否来源于Braess导论所描述的情况。

（2）请你建立模型以分析：如果司机广泛使用可以反映当前交通拥堵情况的GPS导航系统，是否会缓解交通堵塞，并请估计其效果。

# 二、北京市二环路及相关道路概况

二、北京市二环路及相关道路概况北京市二环路是北京市第一条环城快速公路，于 1992 年 9 月建成通车，它是我国第一条全封闭、全立交、没有红绿灯的城市快速环路。二环路处于北京道路路网的核心位置，围绕旧城而建，全长 32.7 公里。沿线共建朝阳门桥、建国门桥、东便门桥、广渠门桥、光明桥、左安门桥、玉蜓桥、永定门桥、陶然桥、右安门桥、菜户营桥、广安门桥、天宁寺桥、复兴门桥、阜成门桥、官园桥、西直门桥、积水潭桥、安定门桥、雍和营桥、东直门桥、东四十条桥等关键键的交通枢纽。具体路线见图 1。

# 三、北京市二环路的Braess 悖论模型

为了考查北京二环路的拥堵是否和Braess 悖论有关，我们先考虑一个简单的Braess网络模型。

# 1基于北京市二环路的Braess 悖论网络模型

# 1.1Braess 悖论网络模型

1 基于北京市二环路的 Braess 悖论网络模型1.1 Braess 悖论网络模型我们假设北京市二环路的客流量均以文慧门为行程起点进入，以左安门作为行程终点，并且在中途不会经过其他匝道下二环。司机们现有两种走法，一种是逆时针行车，经复兴门到达左安门离开二环线，别一种是顺时针行车经建国桥到达左安门。若选择第一种路线，则文慧门到复兴门的距离约为 5km，复

兴门至左安门的距离约为11km。若选择第二条路线，则文慧门到建国桥的距离

图1北京市二环路交通流量图

为  $12\mathrm{km}$ ，从建国桥到左安门的距离为4km。现在假设行车时，在某一段的行驶时间与当时处于该段道路上的车流量成正比。这样，可以假设在文慧门至复兴路段和建国桥至左安门段的行驶时间均为  $\mathsf{a}_1 + \mathsf{b}_1\mathsf{n}_1$  其中  $\mathbf{n}_1$  为当时处于该路段的车辆数目。同样，假设文慧门至建国桥段和复兴门至左安门段的平均行驶时间为  $\mathsf{a}_2 + \mathsf{b}_2\mathsf{n}_2$ ，其中  $\mathbf{n}_2$  为当时处于该路段的车辆数目。可以得到以下的Braess网络模型图：

图1北京市二环线示意图（匝道不开放）

图3北京市二环线示意图（匝道开放）

现在的问题是：

第一，假设有N辆汽车从文慧门开往左安门，如何在二环上分配车辆才能使每辆汽车的运行时间最短？（如图2）

第二，如果在复兴门至建国桥的匝道开放，可以在复兴门经匝道下二环线，选择经前门东大街到建国桥（距离为7km），再上匝道回到二环，从建国桥至左安门，情况又会如何？（如图3）

第一个问题比较简单，我们可以建立以下的线性最优化模型：

为方便起见，我们可以将文慧门简记为W，建国桥简记为J，复兴门简记为F，左安门简记为Z。那么可以得到N辆车的总行驶时间为：

$\mathrm{T} = (\mathbf{a}_1 + \mathbf{b}_1\mathbf{n}_1 + \mathbf{a}_2 + \mathbf{b}_2\mathbf{n}_1)\mathbf{n}_1 + (\mathbf{a}_2 + \mathbf{b}_2\mathbf{n}_2 + \mathbf{a}_1 + \mathbf{b}_1\mathbf{n}_2)\mathbf{n}_2$  ，因此，我们将最小化T作为我们的目标。由此得到一个简单的规划问题。目标函数是

Min  $\mathrm{T} = (\mathbf{a}_1 + \mathbf{b}_1\mathbf{n}_1 + \mathbf{a}_2 + \mathbf{b}_2\mathbf{n}_1)\mathbf{n}_1 + (\mathbf{a}_2 + \mathbf{b}_2\mathbf{n}_2 + \mathbf{a}_1 + \mathbf{b}_1\mathbf{n}_2)$  n约束条件为

s.t.  $\mathrm{n}_1 + \mathrm{n}_2 = \mathrm{N}$

构造Lagrange函数

$$
\mathrm{L}(\mathbf{n}_1,\mathbf{n}_2,\lambda) = (\mathbf{a}_1 + \mathbf{b}_1\mathbf{n}_1 + \mathbf{a}_2 + \mathbf{b}_2\mathbf{n}_1)\mathbf{n}_1 + (\mathbf{a}_2 + \mathbf{b}_2\mathbf{n}_2 + \mathbf{a}_1 + \mathbf{b}_1\mathbf{n}_2)\mathbf{n}_1 + \lambda (\mathbf{N} - \mathbf{n}_1 - \mathbf{n}_2)
$$

求偏得到极值条件为：

$$
\begin{array}{rl} & {\frac{\partial L}{\partial n_1} = \mathbf{a}_1 + \mathbf{a}_2 + 2(\mathbf{b}_1 + \mathbf{b}_2)\mathbf{n}_1\cdot \lambda = 0}\\ & {\frac{\partial L}{\partial n_2} = \mathbf{a}_1 + \mathbf{a}_2 + 2(\mathbf{b}_1 + \mathbf{b}_2)\mathbf{n}_2\cdot \lambda = 0}\\ & {\frac{\partial L}{\partial\lambda} = \mathbf{N}\cdot \mathbf{n}_1\cdot \mathbf{n}_2 = 0} \end{array}
$$

容易解出：

$$
\mathbf{n}_1 = \mathbf{n}_2 = \frac{N}{2}
$$

因此可以看到，此时每辆车的运行时间为：  $\mathrm{t} = \left(\mathrm{a}_{1} + \mathrm{a}_{2}\right) + \left(\mathrm{b}_{1} + \mathrm{b}_{2}\right)\frac{N}{2}$ 。

注意到上述参数中N实际上代表的是道路的长度，道路越长，a值越大，b代表的是道路的宽度，b值越大，道路越宽，路况越好。根据《北京》的数据，我们可以初略地取  $\mathrm{N} = 3600$  。另外，再取  $\mathrm{a}_{1} = 500$ $\mathrm{b}_{1} = 0.03$ $\mathrm{a}_{2} = 1200$ $\mathrm{b}_{2} = 0.02$  ，代入上式，可得：  $\mathrm{n}_{1} = \mathrm{n}_{2} = 1790$  ，即在每条线路上平分车辆才可使得运输效率达到最高，每辆汽车的运行时间约为1790个单位。

# 1.2 Braess 悖论的分析

第二个问题，情况就相对复杂了。为了缓和交通压力，从建国桥到复兴门的匝道开放。因此此时司机有了第三种新的选择，即先选择从文慧门开往复兴门，再由复兴门下匝道，沿前门大街开往建国桥，最后从建国桥上匝道沿二环开往左安门。并且假前门大街由于种种原因仅为单向的开放，并且其路况好于二环，即可以取  $\mathrm{a}_{1} = 700$ $\mathrm{b}_{1} = 0.015$  ，现在我们讨论此种情况下的最优车流。

假设最优时车流的选择线路1：文慧门  $\rightharpoonup$  复兴门  $\rightharpoonup$  左安门的司机有  $\mathbf{x}$  人，选择线路2：文慧门  $\rightharpoonup$  建国桥  $\rightharpoonup$  左安门的司机有  $\mathbf{z}$  人，选择新线路，线路3：文慧门  $\rightharpoonup$  复兴门  $\rightharpoonup$  建国桥  $\rightharpoonup$  左安门的司机有  $\mathbf{y}$  人。那么有：

线路1的司机所花的时间为  $\mathrm{T}_{WFZ} = \mathrm{a}_{1} + \mathrm{b}_{1}(\mathrm{x} + \mathrm{y}) + \mathrm{a}_{2} + \mathrm{b}_{2}\mathrm{x}$

线路2的司机所花的时间为  $\mathrm{T}_{WJZ} = \mathrm{a}_{2} + \mathrm{b}_{2}\mathrm{z} + \mathrm{a}_{1} + \mathrm{b}_{1}(\mathrm{y} + \mathrm{z})$

线路3的司机所花的时间为  $\mathrm{T}_{WFJZ} = \mathrm{a}_{1} + \mathrm{b}_{1}(\mathrm{x} + \mathrm{y}) + \mathrm{a}_{3} + \mathrm{b}_{3}\mathrm{y} + \mathrm{a}_{1} + \mathrm{b}_{1}(\mathrm{y} + \mathrm{z})$  那么所有司机会的总时间，即新情况下的优化目标函数为：

Min  $\mathrm{T} = \mathrm{x}^{*}\mathrm{T}_{WFZ} + \mathrm{y}^{*}\mathrm{T}_{WFJZ} + \mathrm{z}^{*}\mathrm{T}_{WJZ}$

约束条件为：

s.t  $\mathrm{x} + \mathrm{y} + \mathrm{z} = \mathrm{N}$  构造Lagrange函数：

$$
\mathrm{L}\left(\mathrm{x},\mathrm{y},\mathrm{z},\lambda\right) = \mathrm{x}^{*}\left(\mathrm{a}_{1} + \mathrm{b}_{1}\left(\mathrm{x} + \mathrm{y}\right) + \mathrm{a}_{2} + \mathrm{b}_{2}\mathrm{x}\right) + \mathrm{y}^{*}\left(\mathrm{a}_{2} + \mathrm{b}_{2}\mathrm{z} + \mathrm{a}_{1} + \mathrm{b}_{1}\left(\mathrm{y} + \mathrm{z}\right)\right)
$$

$$
+\mathrm{z}^{*}\left(\mathrm{a}_{2} + \mathrm{b}_{2}\mathrm{z} + \mathrm{a}_{1} + \mathrm{b}_{1}\left(\mathrm{y} + \mathrm{z}\right)\right) + \lambda \left(\mathrm{N} - \mathrm{x} - \mathrm{y} - \mathrm{z}\right)
$$

求偏导得到：

$$
\frac{\partial L(x,y,z,\lambda)}{\partial x} = \mathrm{a}_{1} + 2\mathrm{b}_{1}\mathrm{x} + \mathrm{b}_{1}\mathrm{y} + \mathrm{a}_{2} + 2\mathrm{b}_{2}\mathrm{x} + \mathrm{b}_{1} - \lambda = 0
$$

$$
\frac{\partial L(x,y,z,\lambda)}{\partial y} = \mathrm{b}_{1}\mathrm{x} + \mathrm{a}_{1} + \mathrm{b}_{1}(\mathrm{x} + 2\mathrm{y}) + \mathrm{a}_{4} + 2\mathrm{b}_{4}\mathrm{y} + \mathrm{a}_{1} + \mathrm{b}_{1}(2\mathrm{y} + \mathrm{z}) + \mathrm{b}_{1}\mathrm{z} - \lambda = 0
$$

$$
\frac{\partial L(x,y,z,\lambda)}{\partial x} = a_2 + 2b_2z + a_1 + b_1(y + 2z) + b_1y - \lambda = 0
$$

求解，可以得到：

$$
\scriptstyle \mathrm{x} = \mathrm{z} = \frac{2(b_1 + b_4)N + a_4 + (a_2 - a_1)}{2b_1 + 4b_2 + 2b_2}
$$

$$
\scriptstyle \mathrm{y} = \frac{(2b_4 - (b_1 + b_2))N - (a_4 + (a_1 - a_2))}{b_1 + 2b_4 + b_2}
$$

可以看到，在新的情况中，第一种选择和第二种选择的司机人数仍然相等。我们将上面的数据重新代入，可以得到：  $\scriptstyle {\mathrm{x}} = \mathrm{z} = 2250$ $\mathrm{y} = - 900$  。该解看似非常奇怪，但是却是符合逻辑的。结论是：1350辆车走WFZ线，2250辆车沿WJZ线走到建国门（J）之后分道扬，1350辆继续前行到左安门，另外900辆逆行到达目的地。这样，实际上有900辆车走了文慧门  $\rightharpoonup$  建国桥  $\rightharpoonup$  复兴门  $\rightharpoonup$  左安门（WJFZ）的逆行线路。实际此总路线运行的平均时间为2533个单位。这说明，在现在的路网上，由于有前门大街的出现，使得整个交通的运行效率降低，造成了高峰时段二环路的极度拥堵。

# 2基于北京市二环路的Braess网络的动态分析

以上，考虑的是北京市二环路的静态特征，下面从动力学角度分析其动态优化特性。

考虑一个长期的动态均衡过程，即不妨设车辆可以通过自组织达到最优化分配与动态协调的目的。如上节显示，两条不同线路的综合路况相同，那么假设司机选路的原则为前一天选线路1的司机有p的可能性在第二天选择线路2，并记选择第一条线路的司机人数为x，第二条线路的司机人数为z，由此可以得到相应的偏微分方程：

$$
\left\{ \begin{array}{l}\frac{dx}{dt} = -px + pz = p(z - x) \\ \frac{dz}{dt} = -pz + px = p(x - z) \end{array} \right.
$$

因此从微分方程可以明显看出，当且仅当  $\scriptstyle {\mathrm{x}} = \mathrm{z}$  时，有  $\frac{dx}{dt} = \frac{dz}{dt} = 0$  ，从而两种路线的选择达到动态均衡，因此与之前的静态解析解得出结论相同。但是考虑开放匝道后的情况，设选择新线路的司机个数为  $\mathrm{y}$  ，得到以下方程：

$$
\left\{ \begin{array}{l}\frac{dx}{dt} = ax + byz + cz \\ \frac{dy}{dt} = cy - byz - exy \\ \frac{dz}{dt} = cx + exy + fz \end{array} \right.
$$

其中a,b,c,d,e,f均为参数，即在原方程中加入了交叉项的影响。可以原点

会自动记录每条公路上当前的车辆数、汽车经过每条公路平均的占用时间以及它的方差、从起点到终点经过的最长时间、所有汽车从起点到终点的加总时间等统计量，以便讨论。

图4仿真程序示意图

# 3仿真结果及分析

我们共分三种情况进行计算机模拟仿真。第一种是在无GPS导航，车辆在不同分流比例下的行驶情况。第二种是在GPS导航模式一，即认为个人最优，即选择线路3为指导路线时的行驶状况。并且我们用自适应比表示未安装GPS的司机的比例。第三种是在GPS导航模式二，即从全局最优，认为不选择线路3而平均地选择线路1和2为指导路线时的行驶状况。


# 表1情况一的平均行驶时间

表2情况一的道路停留时间方差  


表3情况二的平均行驶时间  


表4情况二的道路停留时间方差  


表5情况三的平均行驶时间  



将各表的平均时间绘制成图如下：

表6情况三的道路停留时间方差 图5情况一的平均行驶时间图

图6情况二的平均行驶时间图

图7情况三的平均行驶时间图

根据表中分析的结果，可得到关于交通网络自适应系统的“Braess 悖论”的主要结论有三点：

（1）当所有车辆都不使用GPS导航系统时，在交通拥堵的高峰时段，尽管许多东西走向的道路提供了更多车辆的路径，但各车辆在选择自身最优路径的同时加剧了其他道路的拥堵，开辟一条新道路反而降低了交通系统的使用效率，存在“Braess 悖论”中描述的问题。

这主要是因为各车辆在各个交通交叉路口都按照自身可观察到的道路状况选择当前最畅通的道路通行，造成了交通网络中的外部性，各车辆通行时不用对交通堵塞和其他车辆的延误付出成本，各车辆最优化自身利益会造成负的外部性，从而无法达到全社会的 Pareto 最优，

如表1所示，当没有GPS导航系统时，表中给出的几种车流分布形式下，

所有车流以相同的比例通过路径1和路径2，而不使用路径3时，全社会的效应达到最优，即所有车辆通过整条道路的时间最短（20217s）。而所有车流采用等概率通过路径1、路径2和路径3是次优解。这种结果所花费的时间约为（22474s），虽没有最优情况下的线路效率高，但比其他车流分布形式通过道路所耗时间要短些。从表中还可看出，当各道路车流分布较为均匀时，全体车辆通过的耗时较短，如车流以  $30\%$  、  $20\%$  、  $50\%$  的比例通过路径1、2、3时所用时间为22620s。相反，如果车流分布趋于极端，即所有的车辆集中于某一条道路上而其他道路处于闲置，则大量车辆会在主干道上拥堵而极大延长通行时间。

（2）当所有车辆广泛使用反映路面状况的GPS导航系统时，GPS系统为司机提供最畅通的路径作为司机参考GPS导航提供的行驶建议，而通过GPS导航系统只是使各车辆掌握整体路况，没有使全体车辆按照社会最优的路线行驶，即以  $50\%$  、  $50\%$  的比例通过路径1、路径2，而放弃新开通的路径3。

通常情况下，司机如果使用GPS导航系统，在决定行驶路径时会参考GPS系统提供的实时路况，但一般也不会完全遵循GPS指定的路径驾驶，而是根据自身驾驶经验及对周围道路现状的观察，将GPS的电子数据和亲身经历结合起来，指定最优最省时的行车路线。基于这样的思路，我们进行的模拟仿真赋予各车辆按照GPS导航行驶的比例，即司机虽然了解了GPS导航的行车建议，但因自身经验、现时观察等多种原因对这一建议并不采纳，而随机选择其他路径。假设这一比例从  $0\%$  、  $25\%$  、  $50\%$  、  $75\%$  、  $100\%$  逐步增加进行模拟仿真，并将各路段所耗平均时间和所有车辆总耗时列于表3中。

根据数据结果，我们发现随着司机对GPS导航系统的接受程度不断增加，由于GPS导航系统也只是替司机提供最优的行车路径，和完全依靠司机自身做决定的情况相比，并没有从实质上改变作最优决策的主体，因此也无法避免“Braess”悖论谈及的效率损失。

（3）当所有车辆广泛使用反映路面状况的GPS导航系统时，GPS系统为司机提供最畅通的路径作为司机参考GPS导航提供的行驶建议并结合自身经验和观察而选择当时条件下的最优路径，两者相互作用的结果会使全体车辆通过道路的时间经过自适应调节，当只有部分车辆听从GPS的建议（表2中为  $10\%$  ）时，全体车辆的行驶时间达到最短，此时全部车辆以  $40\%$  、  $45\%$  、  $10\%$  的比例通过路径1、2、3。随后，如果有多余  $10\%$  的司机完全依据GPS导航提出的路线行驶，则会引导越来越多的车辆向路径3行驶，而在路径1和路径2的主干道上发生越来越严重的拥堵，从而导致全社会效率的降低，车辆通过道路到达目的地的总时间越来越长。这与北京二环在高峰时段发生拥堵的另一个可能的重要原因，即使存在GPS导航也无法使全体车辆的运行达到社会最优。

# 五 结论

本文用随机线性及非线性动态规划方法研究北京二环高峰时段交通堵塞的原因，并用自适应系统的模拟仿真实验验证理论模型。主要结果表明：二环路的道路拥堵符合“Braess悖论”描述的情况，交通堵塞问题并不会因司机广泛使用GPS导航系统而得到缓解，这一结论在细胞自适应系统的仿真模拟实验中得到验证。所以，“Braess”悖论是否发生不仅和个体是否了解全局状况有关，而且还与司机是否遵循GPS建议的最优路线行驶密切相关。
