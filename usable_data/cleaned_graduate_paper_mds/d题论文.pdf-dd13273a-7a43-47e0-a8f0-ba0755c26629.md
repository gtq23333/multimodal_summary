本文讨论了仓库容量有限条件下的随机存贮管理优化问题，认为时间是连续分布的。对于存贮一种商品的问题，根据订货点和自己仓库容量的关系分两种情况讨论，得到平均损失费和订货点及到货时间的关系式，利用实测数据拟合出到货时间的概率密度，使用matlab数学软件解出以平均损失费用的数学期望为目标函数的最优化问题，得到三种商品的最优订货点分别为41，37和36。经过分析得知销售速率与仓库容量的比例，单位商品的损失费对确定订货点都有重要影响。对于存贮多种商品的问题，根据到货时间的取值范围与两个时间临界点（销售完租借仓库中的商品的时间和销售完所有商品的时间）之间的位置关系，将每种商品的到货时间分为六种状态，而对于  $m$  种商品的总损失费用而言，对  $6^{m}$  种不同的组合状态求使得平均总损失费最小的最优订货点和仓库的分配策略，并取使平均总损失费最小的参数为最优化条件。用所建立的模型对给出的三种商品的情形进行求解，得到最优解  $(L^{*},Q_{1},Q_{2},Q_{3},Q_{01},Q_{02},Q_{03})$  为(7.8,3,3,4,3,3,0)。最后，对销售速率随机的情形建立模型并进行了讨论。

对销售速率随机的情形建立模型并进行了讨论。

##############

工厂生产需定期地定购各种原料，商家销售要成批地购进各种商品，这些都涉及到一个怎样存贮的问题。存得少了，无法满足需求，影响利润；存得太多，存贮费用就高。在这种问题中，涉及到交货时间和商品销售速率，需要确定仓库的存贮策略和订货时间以减小总损失费用。

在仓库容量有限的条件下，仓库的存贮策略、订货时间都对总损失费用有一定的影响，交货时间和商品的销售速率多是随机变量。因此，可以在仓库容量和单位商品存贮费及缺货损失费一定的条件下，建立数学模型，求最优订货点和仓库分配以使总损失费最低，并由实际数据估计的随机变量的分布解决实际问题。

# 二、模型假设

假设1商品的销售速率不变（问题1- 4）；

假设2货物到达后仓库中的存货量立即补为  $Q$  ，即卸货时间忽略不计；

假设3时间的连续性：随时检查仓库中的存贮量以确定订货点和计算损失费；假设4先销售租借仓库中的商品；

# 三、变量及符号说明

$q_{2}(t)$  ：时刻t自己的仓库存贮的商品量；

$q_{3}(t)$  ：时刻t租借的仓库存贮的商品量；

$T_{2}$  ：存贮在租借的仓库的商品售完的时间；

$T_{1}$  ：售完所有存贮的商品的时间；

$T$  ：本次到货时间与下次到货时间间隔；

$L$  ：订货时存贮的商品量；

$L_{i}$  ：订货时第  $i$  种商品的存贮体积，  $i = 1,2,\dots,m$  ，  $L_{i}< 0$  时认为该商品已经缺货；

其它变量如题目中所述，个别引入符号在文中会有具体说明。

四、模型的建立、求解与分析

# （一） 只存贮单一商品的情形

# 1、模型的建立

对于这种情形，不可能是在已经缺货的情况下才开始订货，所以  $0< L\leq Q$  为了便于理解，我们根据  $L$  和  $Q_{0}$  的关系分两种情况讨论，建立模型。

(1)  $0< L\leq Q_{0}$  的情况：

一个订货周期  $T$  内总损失费用  $Y$  为  $L$  和  $X$  的函数，即

$$
Y_{1} = \left\{ \begin{array}{l l}{C_{1} + C_{2}*\int_{0}^{T}q_{2}(t)d t + C_{3}*\int_{0}^{T_{2}}q_{3}(t)d t} & {\qquad (X< \frac{L}{r})}\\ {C_{1} + C_{2}*\int_{0}^{T_{1}}q_{2}(t)d t + C_{3}*\int_{0}^{T_{2}}q_{3}(t)d t + C_{4}*\int_{T_{1}}^{T}|q_{2}(t)|d t} & {\qquad (X\geq \frac{L}{r})} \end{array} \right. \tag{1}
$$

其中：

$$
T = \frac{Q - L}{r} +X,T_{1} = \frac{Q}{r},T_{2} = \frac{Q - Q_{0}}{r}
$$

$$
q_{2}(t) = \left\{ \begin{array}{ll}Q_{0} & (0\leq t\leq \frac{Q - Q_{0}}{r})\\ -rt + Q & (\frac{Q - Q_{0}}{r} < t< T) \end{array} \right.,q_{3}(t) = -rt + (Q - Q_{0})\quad (0\leq t< \frac{Q - Q_{0}}{r})
$$

一个订货周期  $T$  内，平均每天的损失费用  $Y(L,X) = \frac{Y}{T}$  ，即

$$
Y_{1}(L,X) = \left\{ \begin{array}{ll}\frac{C_{1}}{T} +\frac{C_{2}}{T} *\int_{0}^{T}q_{2}(t)dt + \frac{C_{3}}{T} *\int_{0}^{T_{2}}q_{3}(t)dt & (X< \frac{L}{r})\\ \displaystyle \frac{C_{1}}{T} +\frac{C_{2}}{T} *\int_{0}^{T_{1}}q_{2}(t)dt + \frac{C_{3}}{T} *\int_{0}^{T_{2}}q_{3}(t)dt + \frac{C_{4}}{T} *\int_{T_{1}}^{T}|q_{2}(t)|dt & (X\geq \frac{L}{r}) \end{array} \right. \tag{2}
$$

代入  $T,T_{1},T_{2},q_{2}(t),q_{3}(t)$  得到：

$$
Y_{1}(L,X) = \left\{ \begin{array}{ll}\frac{C_{1}r}{Q - L + rX} +\frac{C_{3}}{2}\frac{(Q - Q_{0})^{2}}{Q - L + rX} -\frac{C_{2}r}{2}\left(\frac{Q - L}{r} +X\right) + C_{2}Q + \\ \frac{C_{2}}{2}\frac{(Q - Q_{0})^{2}}{Q - L + rX} & (X< \frac{L}{r})\\ \frac{C_{1}r}{Q - L + rX} +\frac{C_{3}}{2}\frac{(Q - Q_{0})^{2}}{Q - L + rX} +\frac{C_{2}}{2}\frac{2QQ_{0} - Q_{0}^{2}}{Q - L + rX} +\\ \frac{C_{4}}{2} [r(\frac{Q - L}{r} +X) - 2Q + \frac{Q^{2}}{Q - L + rX} ] & (X > \frac{L}{r}) \end{array} \right.\dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots
$$

设  $f(x)$  是  $X$  的概率密度函数，  $p_j = P(X = x_j),j = 1,2,\ldots$  是  $X$  的分布律，则  $EY_{1}(L,X) = \int_{- \infty}^{+\infty}Y_{1}(L,x)f(x)dx$  (4) 或者  $EY_{1}(L,X) = \sum_{j}Y_{1}(L,x_{j})p_{j}$  (5)

得到最优化模型： min  $EY_{1}(L,X)$

(2)  $Q_{0}< L< Q$  的情况：

总损失费用为：

$$
Y_{2} = \left\{ \begin{array}{ll}C_{1} + C_{3}\int_{0}^{T}q_{3}(t)dt + C_{2}\int_{0}^{T}Q_{0}dt & (0< X\leq \frac{L - Q_{0}}{r})\\ C_{1} + C_{3}\int_{0}^{T_{2}}q_{3}(t)dt + C_{2}\int_{0}^{T}q_{2}(t)dt & (\frac{L - Q_{0}}{r} < X\leq \frac{L}{r})\\ C_{1} + C_{3}\int_{0}^{T_{3}}q_{3}(t)dt + C_{2}\int_{0}^{T}q_{2}(t)dt + C_{4}\int_{T_{1}}^{T}\left|q_{2}(t)\right|dt & (X > \frac{L}{r}) \end{array} \right.\dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \end{array}
$$

平均每天的损失费用为

$$
Y_{2}(L,X) = \left\{ \begin{array}{ll}\frac{C_{1}}{T} +\frac{C_{3}}{T}\int_{0}^{T_{2}}q_{3}(t) + \frac{C_{2}}{T}\int_{0}^{T}Q_{0}dt & (0< X\leq \frac{L - Q_{0}}{r})\\ \displaystyle \frac{C_{1}}{T} +\frac{C_{3}}{T}\int_{0}^{T_{2}}q_{3}(t) + \frac{C_{2}}{T}\int_{0}^{T}q_{2}(t)dt & (\frac{L - Q_{0}}{r} < X\leq \frac{L}{r})\\ \displaystyle \frac{C_{1}}{T} +\frac{C_{3}}{T}\int_{0}^{T_{2}}q_{3}(t)dt + \frac{C_{2}}{T}\int_{0}^{T_{1}}q_{2}(t)dt + \frac{C_{4}}{T}\int_{T_{1}}^{T}\left|q_{2}(t)\right|dt & (X > \frac{L}{r}) \end{array} \right.\dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots \dots
$$

代入  $T,T_{1},T_{2},q_{2}(t),q_{3}(t)$  得：

$$
Y_{2}(L,X) = \left\{ \begin{array}{ll}\frac{C_{1}r}{Q - L + rX} +\frac{C_{3}}{2} (Q + L - 2Q_{0} - rX) + C_{2}Q_{0} & (0< X\leq \frac{L - Q_{0}}{r})\\ \displaystyle \frac{C_{1}r}{Q - L + rX} +\frac{C_{3}}{2}\frac{(Q - Q_{0})^{2}}{Q - L + rX} -\frac{C_{2}r}{2} (\frac{Q - L}{r} +X) + \\ \displaystyle C_{2}Q + \frac{C_{2}}{2}\frac{(Q - Q_{0})^{2}}{Q - L + rX} & (\frac{L - Q_{0}}{r} < X\leq \frac{L}{r})\\ \displaystyle \frac{C_{1}r}{Q - L + rX} +\frac{C_{3}}{2}\frac{(Q - Q_{0})^{2}}{Q - L + rX} +\frac{C_{2}}{2}\frac{2QQ_{0} - Q_{0}^{2}}{Q - L + rX} +\\ \displaystyle \frac{C_{4}}{2} (L^{2} + r^{2}X^{2} - 2rLX) & (X > \frac{L}{r}) \end{array} \right. \tag{8}
$$

得到最优化模型： min  $EY_{2}(L,X)$

对于某一种商品，分别比较  $\min EY_{1}(L,X)$  和  $\min EY_{2}(L,X)$ ，取二者中较小者对应的  $L^{*}_{j}$  为该商品的最优订货点。

# 2、模型求解与分析

以下代入问题二所给的具体数据分别求解这三种商品的最优订货点。商品一：

将  $r$ ， $c_{1}$ ， $c_{2}$ ， $c_{3}$ ， $c_{4}$ ， $Q_{0}$ ， $Q$  代入（3）式，得：

$$
Y_{1}(L,X) = \left\{ \begin{array}{ll}\frac{140 - 0.005L^{2} + 0.12XL - 0.72X^{2}}{60 - L + 12X} & (X< \frac{L}{r})\\ \displaystyle \frac{140 + 0.475L^{2} - 11.4XL + 68.4X^{2}}{60 - L + 12X} & (X\geq \frac{L}{r}) \end{array} \right. \tag{9}
$$

根据题中所述以及对所给数据的分析，我们认为  $X$  可以是0—7天中的某个随机天数，而且每个数字应该表示一段时间，比如数字7表示的是时间轴上的区间[7,8)。所以在求概率密度时进行了对  $X$  的频率从0到8的拟合。

用origin中的Lorentz函数拟合  $X$  的概率密度函数，得

$$
f(x) = \left\{ \begin{array}{ll}\frac{0.009702 + \frac{2}{\pi 4^{*}(x - 2.9832)^{2} + 1.5138^{2}}} & (0\leq x< 8)\\ 0 & (x< 0\exists x\geq 8) \end{array} \right. \tag{10}
$$

用matlab求解式(4)，并用选带法求得  $\min EY_{1}(L,X) = 3.5887$ ，此时  $L^{*}_{1} = 39.9999$ 。

将  $r$ ， $c_{1}$ ， $c_{2}$ ， $c_{3}$ ， $c_{4}$ ， $Q_{0}$ ， $Q$  代入（8）式，则有：

$$
Y_{2}(L,X) = \left\{ \begin{array}{ll}\frac{132 + 0.4L - 4.8X + 0.24LX - 0.01L^{2} - 1.44X^{2}}{60 - L + 12X} & (0< X\leq \frac{L - Q_{0}}{r})\\ \displaystyle \frac{140 - 0.005L^{2} + 0.12XL - 0.72X^{2}}{60 - L + 12X} & (\frac{L - Q_{0}}{r} < X\leq \frac{L}{r})\\ \displaystyle \frac{140 + 0.475L^{2} - 11.4XL + 68.4X^{2}}{60 - L + 12X} & (X > \frac{L}{r}) \end{array} \right.
$$

求解式(4)，并求得  $\min EY_{2}(L,X) = 3.5831$ ，此时  $L^{*}_{2} = 41.3918$

# 商品二：

利用  $Y_{1}(L,X)$  求得  $\min EY_{1}(L,X) = 4.2252$ ， $L^{*}_{1} = 37.0612$  利用  $Y_{2}(L,X)$  求得  $\min EY_{2}(L,X) = 4.2636$ ， $L^{*}_{2} = 40.0000$

# 商品三：

利用  $Y_{1}(L,X)$  求得  $\min EY_{1}(L,X) = 11.6191$ ， $L^{*}_{1} = 20.0000$  利用  $Y_{2}(L,X)$  求得  $\min EY_{2}(L,X) = 9.6367$ ， $L^{*}_{2} = 36.4637$

根据以上计算，得到三种商品的最优订货点分别取为41.3918，37.0612，36.4637，考虑到实际情况，分别取为41，37，36。

结合这三种商品的数据及计算结果，可以看出，我们建立的模型比较合理，适用于解决一些实际问题。当单位商品的各种损失费用相同时，对于销售速率较小且仓库容量较大的商品，订货点较小，因为这类商品的缺货风险较小，而损失主要是由存贮造成的；反之订货点较大，因为该类商品的缺货风险较大，损失主要由缺货造成。但是由于不同商品各单位损失费用有差别，而该费用对商品订货点也有一定影响。一般来说，单位商品的存贮费用越高，订货点就应该越小；而单位商品的缺货费用越高，订货点应该越大。

# (二) 存贮  $m$  种商品的情形

# 1、模型的建立

在一个订货周期  $T$  中，设  $X\in [a,b]$ ， $(0\leq a< b)$ ，即交货时间  $X$  只能在这个区间。根据区间  $[a,b]$  与两个临界点  $\frac{Q_{i} - Q_{0i}}{r_{i}\nu_{i}}$ （销售完租借仓库中的商品的时间）和  $\frac{Q_{i}}{r_{i}\nu_{i}}$ （销售完所有商品的时间）之间的位置关系，这  $m$  种商品的到货时间分别存在六种状态：①当  $0\leq T_{L} + a< T_{L} + b\leq \frac{Q_{i} - Q_{0i}}{r_{i}\nu_{i}}$  时，

$$
\begin{array}{c}{Y_{i1}(X,L_i,Q_i,Q_{0i}) = \frac{1}{2T} C_{3i}(T_L + X)[(Q_i - Q_{0i}) + Q_i - (T_L + X)r_i\nu_i - Q_{0i}] + }\\ {\frac{C_{2i}}{T} Q_{0i}(T_L + X)} \end{array} \tag{12}
$$

②当  $\frac{Q_i - Q_{0i}}{r_i\nu_i} < T_L + a < T_L + b \leq \frac{Q_i}{r_i\nu_i}$  时，

$$
\begin{array}{c}{Y_{i2}(X,L_i,Q_i,Q_{0i}) = \frac{1}{T}\frac{C_{3i}}{2} (Q_i - Q_{0i})\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{C_{2i}}{T} Q_{0i}\frac{Q_i - Q_{0i}}{r_i\nu_i} +}\\ {\frac{C_{2i}}{2T} (T_L + X - \frac{Q_i - Q_{0i}}{r_i\nu_i})[Q_{0i} + Q - (T_L + X)r_i\nu_i]} \end{array} \tag{13}
$$

③当  $\frac{Q_i}{r_i\nu_i} < T_L + a < T_L + b$  时，

$$
\begin{array}{c}{Y_{i3}(X,L_i,Q_i,Q_{0i}) = \frac{C_{3i}}{2T} (Q_i - Q_{0i})\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{C_{2i}}{2T} Q_{0i}\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{Q_i}{r_i\nu_i} +}\\ {\frac{C_{4i}}{2T} (T_L + X - \frac{Q_i - Q_{0i}}{r_i\nu_i})(T_L + X - \frac{Q_i}{r_i\nu_i})r_i\nu_i} \end{array} \tag{14}
$$

④当  $T_L + a < \frac{Q_i - Q_{0i}}{r_i\nu_i} < T_L + b \leq \frac{Q_i}{r_i\nu_i}$  时，

$$
Y_{4i}(X,L_i,Q_i,Q_{0i}) = \left\{ \begin{array}{ll}\frac{1}{2T} C_{3i}(T_L + X)[(Q_i - Q_{0i}) + Q_i - (T_L + X)r_i\nu_i - Q_{0i}] + \\ \qquad C_{2i}Q_{0i}(T_L + X) & (T_L + X\leq \frac{Q_i - Q_{0i}}{r_i\nu_i})\\ \frac{1}{T}\frac{C_{3i}}{2} (Q_i - Q_{0i})\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{C_{2i}}{T} Q_{0i}\frac{Q_i - Q_{0i}}{r_i\nu_i} +\\ \frac{C_{2i}}{2T} (T_L + X - \frac{Q_i - Q_{0i}}{r_i\nu_i})[Q_{0i} + Q - (T_L + X)r_i\nu_i] & (T_L + X > \frac{Q_i - Q_{0i}}{r_i\nu_i}) \end{array} \right. \tag{15}
$$

⑤当  $\frac{Q_i - Q_{0i}}{r_i\nu_i} \leq T_L + a < \frac{Q_i}{r_i\nu_i} < T_L + b$  时，

$$
Y_{5i}(X,L_i,Q_i,Q_{0i}) = \left\{ \begin{array}{ll}\frac{1}{T}\frac{C_{3i}}{2} (Q_i - Q_{0i})\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{C_{2i}}{T} Q_{0i}\frac{Q_i - Q_{0i}}{r_i\nu_i} +\\ \frac{C_{2i}}{2T} (T_L + X - \frac{\bar{Q}_i - Q_{0i}}{r_i\nu_i})[Q_{0i} + Q - (T_L + X)r_i\nu_i] & (T_L + X\leq \frac{Q_i}{r_i\nu_i})\\ \frac{C_{3i}}{2T} (Q_i - Q_{0i})\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{C_{2i}}{2T} Q_{0i}(\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{Q_i}{r_i\nu_i}) + \\ \frac{C_{4i}}{2T} (T_L + X - \frac{Q_i - Q_{0i}}{r_i\nu_i})(T_L + X - \frac{Q_i}{r_i\nu_i})r_i\nu_i & (T_L + X > \frac{Q_i}{r_i\nu_i}) \end{array} \right. \tag{16}
$$

⑥当  $T_L + a < \frac{Q_i - Q_{0i}}{r_i\nu_i}$  且  $T_L + b > \frac{Q_i}{r_i\nu_i}$  时，

$$
Y_{6i}(X,L_i,Q_i,Q_{0i}) = \left\{ \begin{array}{ll}\frac{1}{2T} C_{3i}(T_L + X)[(Q_i - Q_{0i}) + Q_i - (T_L + X)r_i\nu_i - Q_{0i}] + \\ \qquad C_{2i}Q_{0i}(T_L + X) & (T_L + X\leq \frac{Q_i - Q_{0i}}{r_i\nu_i})\\ \frac{1}{T}\frac{C_{3i}}{2} (Q_i - Q_{0i})\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{C_{2i}}{T} Q_{0i}\frac{Q_i - Q_{0i}}{r_i\nu_i} + & \dots (17)\\ \frac{C_{2i}}{2T} (T_L + X - \frac{Q_i - Q_{0i}}{r_i\nu_i})[Q_{0i} + Q - (T_L + X)r_i\nu_i] & (\frac{Q_i - Q_{0i}}{r_i\nu_i} < T_L + X\leq \frac{Q_i}{r_i\nu_i})\\ \frac{C_{3i}}{2T} (Q_i - Q_{0i})\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{C_{2i}}{2T} Q_{0i}(\frac{Q_i - Q_{0i}}{r_i\nu_i} +\frac{Q_i}{r_i\nu_i}) + \\ \frac{C_{4i}}{2T} (T_L + X - \frac{Q_i - Q_{0i}}{r_i\nu_i})(T_L + X - \frac{Q_i}{r_i\nu_i})r_i\nu_i & (T_L + X > \frac{Q_i}{r_i\nu_i}) \end{array} \right. \tag{17}
$$

每一种商品都存在这六种可能的状态，因此就  $m$  种商品总损失费的整体而

言，需要对  $6^{m}$  种不同状态分别求使得  $EY_{j}(X,L_{i},Q_{i},Q_{0i}) = \frac{\sum_{i = 1}^{m}Y_{i} + C_{1}}{T}$ $(j = 1,2,\dots,6^{m})$  最小的解  $L_{i}^{*},Q_{0i}^{*},Q_{i}^{*}$

即建立最优化模型：min  $EY_{j}(X,L_{i},Q_{i},Q_{0i})$

$$
\begin{array}{r}\left\{ \begin{array}{ll}\sum_{i = 1}^{m}Q_{0i} = Q_{0},\\ \displaystyle \sum_{i = 1}^{m}Q_{i} = Q,\\ \displaystyle T_{L} = \frac{Q_{i} - L_{i}}{\nu_{i}r_{i}} = \frac{Q_{j} - L_{j}}{\nu_{j}r_{j}}\quad (i\neq j),\\ \displaystyle Q_{0i}\geq 0,Q_{i} > 0,Q_{i}\geq Q_{0i},Q_{i}\geq L_{i}, \end{array} \right. \end{array} \tag{s.t.}
$$

每种状态下的最优订货点为  $L_{j}^{*} = \sum_{i = 1}^{m}L_{i}(L_{i} > 0)$

比较  $\min EY_{j}(X,L_{i},Q_{i},Q_{0i})$  ，则这  $6^{m}$  种结果中优化值最小者对应的 $L^{*}_{k},Q_{k},Q_{k_{0i}}$  为问题的最优化条件。即若  $\min EY_{k}(X,L_{i},Q_{i},Q_{0i})<$ $\min EY_{s}(X,L_{i},Q_{i},Q_{0i})$ $(s = 1,2,\dots,6^{m}$  且  $s\neq k)$  ，则  $EY_{k}(X,L_{i},Q_{i},Q_{0i})$  所对应的 $L^{*}_{k},Q_{k},Q_{0k}$  即为问题的最优解。

# 2、模型求解与分析

对于问题四所给商品的情形，我们首先对这  $6^{3}$  种可能的状态组合进行分析，根据所给数据之间的关系，排除了那些不可能产生最优总损失费用的状态。只对其余的组合状态用 matlab 分别求出目标函数的最优值及其所对应的最有解

$L_{1},L_{2},L_{3},Q_{1},Q_{2},Q_{3},Q_{01},Q_{02},Q_{03}$ ，然后去除标记符号为- 1(表示无最有解，而计算程序所给出的为近似的最小二乘解)的情况，最后得到如表一所示的结果(已按照目标函数值大小排列)。比较出最小的平均损失费为组合状态为(226)时的目标函数值，相应的最优解为  $(2,4,2,4,2,3,3,4,3,3,0)$ ，所以  $L^{\ast} = 7.8$ 。

表一三种商品可能出现最优解的状态组合及对应的函数最优值  

<table><tr><td>序号</td><td>组合状态</td><td>目标函数值</td><td>标记符号</td></tr><tr><td>1</td><td>226</td><td>3.1513</td><td>1</td></tr><tr><td>2</td><td>526</td><td>3.1762</td><td>1</td></tr><tr><td>3</td><td>326</td><td>3.1773</td><td>1</td></tr><tr><td>4</td><td>536</td><td>3.2302</td><td>1</td></tr><tr><td>5</td><td>266</td><td>3.6513</td><td>1</td></tr><tr><td>6</td><td>246</td><td>3.6873</td><td>1</td></tr><tr><td>7</td><td>626</td><td>3.7356</td><td>1</td></tr><tr><td>8</td><td>546</td><td>3.7423</td><td>1</td></tr><tr><td>9</td><td>346</td><td>3.7529</td><td>1</td></tr><tr><td>10</td><td>566</td><td>3.7866</td><td>1</td></tr><tr><td>11</td><td>636</td><td>3.8339</td><td>1</td></tr><tr><td>12</td><td>366</td><td>3.8512</td><td>1</td></tr><tr><td>13</td><td>352</td><td>3.9513</td><td>1</td></tr><tr><td>14</td><td>532</td><td>4.1863</td><td>1</td></tr><tr><td>15</td><td>534</td><td>4.1863</td><td>1</td></tr><tr><td>16</td><td>632</td><td>4.1863</td><td>1</td></tr><tr><td>17</td><td>634</td><td>4.1863</td><td>1</td></tr><tr><td>18</td><td>646</td><td>4.2192</td><td>1</td></tr><tr><td>19</td><td>666</td><td>4.2912</td><td>1</td></tr><tr><td>20</td><td>552</td><td>4.6907</td><td>1</td></tr><tr><td>21</td><td>554</td><td>4.6907</td><td>1</td></tr><tr><td>22</td><td>562</td><td>4.6907</td><td>1</td></tr><tr><td>23</td><td>564</td><td>4.6907</td><td>1</td></tr><tr><td>24</td><td>652</td><td>4.6907</td><td>1</td></tr><tr><td>25</td><td>654</td><td>4.6907</td><td>1</td></tr><tr><td>26</td><td>662</td><td>4.6907</td><td>1</td></tr><tr><td>27</td><td>664</td><td>4.6907</td><td>1</td></tr></table>

注：组合状态(ijk)表示这三种商品所处的状态分别为  $i,j,k$  0

# （三）商品的销售是随机的情形

设第  $i$  种商品的损失费用  $Y_{i}$  与  $L_{i}$ ， $Q_{i}$ ， $Q_{0i}$ ， $X$ ， $r_{i}$  的关系式为

$Y_{i} = g(L_{i},Q_{i},Q_{0i},X,r_{i})$ ，随机变量  $X$  与  $r$  的联合概率密度为  $f(X,r)$ ，所以平均每

天的总损失费用为  $Y(X,L,Q,Q_0,r) = \frac{\sum_{i = 1}^{m}Y_i + C_1}{T}$ ，（其中  $Q = (Q_{1},Q_{2},\ldots ,Q_{m})^{T}$ ， $Q_{0} = (Q_{01},Q_{02},\ldots ,Q_{0m})^{T}$ ， $L = (L_{1},L_{2},\ldots ,L_{m})^{T}$ ， $r = (r_{1},r_{2},\ldots ,r_{m})^{T}$ ）则其期望为  $EY(X,L,Q,Q_0,r) = E(\frac{\sum_{i = 1}^{m}Y_i + C_1}{T})$ 即  $EY(L,Q,Q_0,X,r) = \int_{- \infty}^{+\infty}\dots \int_{- \infty}^{+\infty}g(L,Q,Q_0,x,r)f(x,r)dxdr$

由此可以建立最优化模型： min  $EY(L,Q,Q_0,X,r)$

$$
\left\{ \begin{array}{l}0< L\leq Q\\ \sum_{i = 1}^{m}Q_{0i} = Q_{0},\\ \sum_{i = 1}^{m}Q_{i} = Q,\\ T_{L} = \frac{Q_{i} - L_{i}}{\nu_{i}r_{i}} = \frac{Q_{j} - L_{j}}{\nu_{j}r_{j}}\quad (i\neq j),\\ Q_{0i}\geq 0,Q_{i} > 0,Q_{i}\geq Q_{0},Q_{i}\geq L_{i} \end{array} \right.
$$

为了降低损失费用，无论是  $X$  分布发生变化还是  $r$  发生变化，都必然导致定货点  $L$  的改变，这一特性在模型中正好明显体现出来。例如，当  $X$  或  $r$  在较大值处的概率密度增加时应该调整  $L^{*}$  使之增大；反之，则应该减小  $L^{*}$ 。

参考文献：

[1]欧俊豪，王家生，徐漪萍，等.应用概率统计[M].天津：天津大学出版社，1999[2]刘承平.数学建模方法[M].北京：高等教育出版社，2002[3]吴翊，吴孟达，成礼智.数学建模的理论与实践[M].长沙：国防科技大学出版社，1999[4]朱道元.数学建模案例精选[M].北京：科学出版社，2003[5]宋兆基，徐刘美.Matlab6.5在科学计算中的应用[M].北京：清华大学出版社，2005
