本文针对轮胎花纹的设计建立了一个多目标规划的模型。通过轮胎花纹对于轮胎性能的影响，我们将所影响轮胎的性能转化为六项可见的指标（承载性能、防滑性能、牵引性能、减噪性能、耐磨性能），并以这六项指标来建立多目标规划的模型，并用TOPSIS分析法来得到最终的最优解。

对于如何设计出轮胎花纹，我们可将其分解为以下几个部分来求解影响花纹设计的几个参数，最终得到轮胎花纹的设计方案。

第一部分：我们将驾车者对于轮胎使用需求分为三类：车辆情况、路面条件、行车条件。将这三类需求的每种情况所对应的轮胎性能的要求进行量化，并用矩阵表示。同时，通过权值分析，将给定车辆情况、路面条件、行车条件后对轮胎性能的要求表示出来，并用储存在目标向量。

第二部分：结合文献资料，我们总结出3个轮胎花纹设计要素（轮胎花纹走向、沟槽比、沟槽深度），并且将花纹的设计因素对轮胎性能的影响进行评价，最终进行量化。通过引入参数  $\theta$  （横纹倾斜度），  $x$  （横纹所占总花纹面积比），  $b$  （沟槽比），  $c$  （沟槽深度），然后由目标向量来建立一个非线性规划模型，再对其进行优化，将非线性规划转化为图中寻求最优路径的问题。

第三部分：在第二部分中寻找到了所有可行路径后，为了寻求在多个目标均最优的条件下最优解，通过TOPSIS分析法，对所有可行路径进行从优到劣的排列，得到所需求的最优路径，从而也确定了花纹的设计方案。

参赛队号： 2011队

所选题目： A 题

![](images/0f6471561b8d5a6375d41f905ec4d1dae88e2d7f49999f03b9d5dfaf7e72be5f.jpg)

参赛队号#2011

This article was aiming at establishing a model of multiple objective programming about tire patterns. We summarized six visible targets (supporting value, slip resistance, buffer performance the value of tractive force, the value of noise reduction, the value of wear- resistance) to set up a model of multiple objective programming via the influence of tire patterns. Finally, we got the optimal solution in the way of TOPSIS analysis.

In order to design tire patterns, the project was divided by three parts. In every part, we got influential factors about tire patterns. Ultimately, we put forward the design of tire patterns.

Part one; we separated the demand of drivers into three sections: the condition of vehicles, earth and driving. Quantizing factors of tire performance, expressed by matrix. At the same time, show the condition of vehicles, earth and driving by the analysis of weight, the stored in objective vectors.

Part two; according to all kinds of articles, we summarized three designing factors of tire patterns (the direction of tire patterns, the rate of tire groove, the depth of groove) and evaluated designing factors of tire patterns, then quantized these factors. Furthermore, we established a model of nonlinear programming in the use of objective vectors via factor (inclination of horizontal lines),x(the rate of horizontal lines,), b(the rate of tire groove), c(the depth of tire groove), and optimize these factors. Finally, transform into the question of searching the optimum paths.

Part three; find all paths of the question, in order to find the best answer among all solutions. We choose the approach of TOPSIS analysis to sequence all feasible paths from the best to the worst, and then got the optimum path. Finally, got the project of tire patterns.

Key words: multiple objective programming, the approach of TOPSIS, nonlinear programming, quantitative analysis

参赛队号#2011

##############

轮胎被广泛使用在多种陆地交通工具上。根据性能的需要，轮胎表面常会加工出不同形状的花纹。在设计轮胎时，往往要针对其使用环境，设计出相应的花纹形状。

通过第一阶段的数学建模，分析不同轮胎花纹对轮胎性能的影响，我们已得到了一个较为完善的轮胎花纹评价体系，第二阶段则是当车辆情况、路面条件和使用需求确定时，设计出合适的轮胎花纹。

然而在可以设计出比较理想合适的轮胎花纹前，我们需要解决这样几个问题：

1. 如何将消费者所给定的车辆情况、路面条件、使用需求定量转化为轮胎花纹性能评价指标，这一步需要定性分析，并且最后需要定量转化为轮胎性能指标。

2. 如何将轮胎花纹分解为几个重要参数的组合，并求得不同参数对轮胎性能的影响，并将将其量化。

3. 如何根据消费者提出的已经被量化的性能指标需求确定轮胎花纹组合区间，最后通过多目标规划的到轮胎花纹设计的最优化组合。

以这三个问题为导向，我们设计出一个轮胎花纹解决方案流程图来辅助我们更好地了解其中的内在联系。

图1-1轮胎花纹设计过程框图

# 第七届数学中国数学建模网络挑战赛


# 二. 问题分析

在解决上述三个问题之前，我们首先确定对轮胎的性能评价分类：承载性能、防滑性能、牵引性能、减噪性能、缓冲性能。耐磨性能，这六项性能基本包括了轮胎能力涵盖的范围。

针对问题1，不同消费者会给出不同的车辆情况、路面条件、使用需求。以路面条件为例，就可以划分为沙地、碎石地、山地、雨雪地、沼泽地、高速公路、沥青路面、水泥路面，这样的分类方式过于繁杂，处理数据过程中很容易出现纰漏。如果仅依靠几个特殊的条件得到的花纹组合那么将不具有解决问题的通用性、很难体现数学建模的实际意义。我们要做的便是通过资料的收集，将消费者对于车辆情况、路面条件、使用需求的约束条件进行归纳分析，将其分为三大类，每个大类选取典型的影响因子，将这些典型的影响因子量化，这样做即达到了简化数学模型的目的，又不会丢失过多的影响因素，影响文章的准确性。

针对问题2，通过对第一阶段问题的研究，我们得到了轮胎花纹的性能特征、影响因素，但这些结论大部分是定性结论，如果想实现给定条件下设计出合适的花纹，必须将花纹设计因素量化成性能评分，通过分数评定得到给定条件下的花纹组合。

通过查询一系列资料，我们将花纹的设计因素归纳为轮胎花纹走向、沟槽比、沟槽深度。将这三个设计因素与轮胎的六项性能建立分值联系，达到量化的效果。其中在花纹走向的分析上，我们将横纵向花纹根据其在整体花纹组合中的贡献度进行复合，得到一个比较完善的花纹走向评分模型；沟槽比、沟槽深度则通过资料介绍、测量得到合理的区间范围，引入量化模型。

针对问题3，通过前两个问题已经分别得到了花纹性能评价的量化评分指标，实际需求条件对于性能的量化评分指标，通过某一给定的实际需求指标，计算出能够满足该需求的轮胎花纹所有组合，最后通过多目标规划等数学建模方法，减小可行域，得到相应的可行解，再通过对于实际问题的分析得到轮胎花纹设计的最优化解决方案，完成轮胎花纹设计方案。

# 三. 问题假设

1. 假设轮胎使用的材质相同；2. 假设轮胎的半径以及胎壁厚度相同；3. 假设轮胎花纹性质仅由花纹走向，沟槽比，沟深决定。

# 四. 符号说明


# 第七届数学中国数学建模网络挑战赛



# 五. 模型准备

# 5.1 驾车者行车条件的确定

5.1.1

针对问题1给定车辆情况、路面条件、行车条件比较宽泛，我们将条件细化，得车辆情况：货车/轿车/客车/越野车/工程车路面情况：山地/沙地/水泥地/沥青路行车条件：速度/舒适度/使用寿命/安全性能这些具体的给定条件对于轮胎的性能要求又不尽相同，

# 5.1.2 车辆类型方面

承重性能便是货车对于轮胎的刚性需求，满足载重量大的同时，还需要轮胎具有良好的防滑性、耐磨性，这样才能够适应长途未知天气运送。相对来说，货车对于轮胎降噪、缓冲的要求并不是特别高，只需要满足最低标准即可；然而对于轿车这类的家用车，驾车体验便是比较重要的一环，私家车要求防滑系数高以保证行车安全，轮胎噪声小、缓冲能力强以提供最佳的行车舒适度，同时要求轮胎比较耐磨，家庭形式的外出对于轿车的载重能力需求不大；对于客车，不需要货车强大的载货能力，但是对于载人承重一定需求，客车乘坐人员众多，因此对于安全性能的要求特别高，防滑能力、制动能力都是重中之重，而缓冲性能、减噪性能、耐磨性能只需要满足国家基础标准即可；越野车辆对于野外地形的适应能力要求很强，这就要求越野车轮胎有出色的防滑能力、翻越山坡的强劲牵引力、应对崎岖路面优秀的缓冲能力、长途旅程适当的耐磨性能；工程车是一个建筑工程的主干力量，常见的工程车辆包括大型吊车、挖掘机、推土机等，这类工程车辆对于轮胎的性能要求比较单一，由于车身的重量在几吨、十几吨左右，因此要求轮胎的承重能力特别好[1]，需要保证较好的牵引力使启动与制动比较灵敏，一定的耐磨能力保证轮胎不易在苛刻的工作环境下迅速损坏，至于其他方面要求并不明显、可以忽略不计。

# 5.1.2 路面情况方面

山地需要车辆具有良好的牵引力，保证车辆在山坡类型的路面有足够的动力翻越，[2]同时还需要车辆优秀的缓冲能力，使其在崎岖路面可以较平稳的行驶；沙地路面则需要车辆载重不能太高，以免陷入砂石中打滑，无法前进；水泥路面、沥青路面都是行驶

# 第七届数学中国数学建模网络挑战赛


条件比较好的路面情况，对于轮胎的要求并不是特别高，只需要注意雨雪天气的防滑能力即可。

# 5.1.3使用需求方面

我们根据消费者车辆使用报告得到，消费者比较看重的几项车辆性能，即速度、舒适度、使用寿命、安全性能。速度则是对于车辆牵引力的需求，若果消费者对于速度方面有需求，那么牵引力性能要达到中等水平以上；舒适度则是对减噪效果与缓冲能力两方面提出了一定的要求；消费者更期望较长的轮胎使用寿命避免频繁更换轮胎，因此耐磨性就是评价使用寿命的重要因素；安全性能要求车辆在遇到紧急情况下可以迅速做出反应，及时刹车，这需要汽车具有良好的制动能力；同时也需要轮胎在雨雪天气下可以有优秀的防滑能力。

我们结合消费者行车习惯、满意度调查报告以及论坛网友们的实际体验，总结出来车辆类型、路面情况、使用需求对于轮胎各项性能需求标准，并以此为依据进项性能评分将消费者的需求转化为性能评分，再与轮胎花纹设计评价模型进行匹配，最后得到给定条件下适合的轮胎花纹。

# 5.2轮胎花纹设计要素的确定

5.2 轮胎花纹设计要素的确定轮胎花纹的设计决定着轮胎的最终性能，根据消费者对轮胎性能的需求，将这些需求转化为相应的性能指标分数，通过非线性规划数学模型算出符合该性能最佳轮胎花纹组合方案。在完成这项评价体系前，我们通过查询大量资料，总结出轮胎花纹3个设计要素，即轮胎花纹走向、沟槽比、沟槽深度。

# 5.2.1轮胎花纹走向

# 5.2.1.1普通花纹

普通轮胎花纹走向分为横向花纹与纵向花纹

横向花纹：其特点是胎面横向连续，纵向断开，因而胎面横向刚度大，而纵向刚度小。故轮胎抗滑能力呈现出纵强横弱，整体防滑能力并不出色[3]；

当轮胎高速运动时，滚动阻力会明显增大，导致轮胎磨损很严重；路况较好的道路高速行驶，横向花纹块之间的间隙比较大，有颠簸感，缓冲性能差；

横向条纹与地面挤压碰撞，产生很大的噪声；

然而横向花纹的排布与整体结构决定轮胎比较坚硬，具有良好的承载能力，因此适用于大型货运类、工程机械类车辆。

优势：横向花纹在启动与制动过程中可以提供强大的牵引力、制动力，同时赋予轮胎优秀的承载性能，但是在防滑性能方面表现一般。

劣势：轮胎在高速行驶时，缓冲效果差，滚动阻力会明显增加，导致噪音很大，轮胎耐磨性能下降。

纵向花纹：其特点是胎面纵向连续，横向断开，因而胎面纵向刚度大，而横向刚度小，轮胎抗滑能力叶现出横强纵弱。这种花纹轮胎的滚动阻力较小，因此牵引力制动力相比横向不足，决定了承载能力较差[4]；

纵向花纹的排布有利于及时排出轮胎间隙中的积水，避免出现滑水现象。纵向花纹高速行驶时滚动阻力较小，耐磨性能增加、噪音减小、缓冲能力加强。

# 第七届数学中国数学建模网络挑战赛


优势：纵向花纹具有低滚动阻力，高速性能较好，噪声较低、耐磨性能优秀、防滑效果好。

劣势：由于牵引力、制动力较差，导致纵向花纹的承载能力差。

5.2.1.2复合轮胎花纹方案：

单一的轮胎花纹很难满足现代人们对于汽车的各项要求，催生复合花纹轮胎。我们小组也致力于给出一个完备的轮胎花纹设计组合，因此我们利用非线性规划的数学方法，根据横纵向花纹对于轮胎性能的贡献程度拟合出一个复合轮胎花纹指标计算公式，该公式在特殊的条件下可以演化成单一花纹的性能评价公式，符合波尔评价论文原则：新推算的公式是旧公式的衍生，且在特殊条件下可简化为已知公式。

# 5.2.2轮胎花纹深度：

根据资料显示，轮胎花纹的深度一般为10—12mm，在使用的过程中轮胎的花纹深度会逐渐减少，当深度过低会产生严重的安全隐患，因此各个国家会设定一个轮胎花纹磨损极限，中国的标准为2mm。在相同的轮胎厚度的前提下，我们将轮胎深度上限定为11mm，下限定位2mm，讨论在此区间内，不同的轮胎花纹深度对轮胎各项性能的影响。

随着轮胎花纹深度的减少：

轮胎空隙率减小，一体性加强，载重能力逐渐加强但缓冲能力会相应的减弱，当轮胎胎纹深度为2mm时，胎纹深度很小，这样的轮胎一般应用于载重工程车辆；

但是花纹深度减少不利于轮胎储水排水，易产生滑水现象，并使光胎面易打滑的弊端显现出来[5]；

花纹块接地的弹性形变量减小，轮胎弹性迟滞损失形成的滚动阻力减小，牵引力、制动力相应减少[6]，当轮胎花纹深度达到8mm时，处在正常轮胎的最佳性能区，牵引力最好，11mm一般为新轮胎的胎纹深度，性能无法完全发挥；

通过论文[7]可知，轮胎深度对于轮胎的噪声影响并不大，故不做深入考虑；

轮胎花纹的耐磨性能与胎纹深度成二次函数关系，较厚的胎纹滚动阻力大，因此损耗导致轮胎易磨损消耗，随着轮胎深度的减少，轮胎滚动阻力减少，磨损减少。当胎纹深度过小时，接近光胎面，轮胎打滑，导致磨损再次加大。

# 5.2.3沟槽比：

沟槽比即为沟槽占整个轮胎表面的百分比，通过实际测量计算得到轿车轮胎花纹沟槽比为0.1—0.2、工程机械车轮胎沟槽比为0.3—0.4；农机类轮胎沟槽比为0.7—0.8。

0.1—0.7的区间包括了绝大多数的车辆轮胎类型。我们便将这个区域定为沟槽比的合理区域。

沟槽比越低，轮胎橡胶与地面接触的面积越大，相应的抓地力、牵引力越强，这使轮胎的承重能力相应地加强；当沟槽比较大时，轮胎接近农机轮胎具有较大的抓地力、牵引力及承重能力；

随着沟槽比的增加，轮胎的储水、排水效果加强，防滑能力得到提升；

沟槽比增加将导致轮胎空隙加大，滚动阻力加大，耐磨性能减弱，轮胎噪声增加[9]；

# 六.模型建立与求解

# 第七届数学中国数学建模网络挑战赛


# 6.1驾车者使用需求量值化模型的确立

基于上材料，我们可以将车辆情况、路面条件和行车条件归结为以下几种情况：

表6-1车辆类型的分类  


而对于以上的不同情况，对于轮胎的各种性能的要求又是不同的，同样的，对于我们所收集的资料中总结出各种不同的情况对于轮胎性能要求指标：

表6-2车辆类型要求轮胎最低性能  


表6-3路面情况要求轮胎最低性能  


表6-4使用需求要求轮胎最低性能  


由上述的表格中，我们可以将消费者对于车辆的使用需求转化为对轮胎性能的定量数值要求。

# 6.2花纹子模块对轮胎性能指标评价

6.2.1花纹子模块对轮胎性能影响的量化模型

由问题分析的材料中，我们总结出花纹走向、花纹沟槽比、花纹沟深对轮胎的六种性能值的影响，详见下表：

# 第七届数学中国数学建模网络挑战赛

表6-5花纹子模块的评价表

表6-6花纹子模块的量化数值表  


在上表中，我们用优秀，良好，中与差四个指标来衡量花纹走向、沟槽比和沟深对于轮胎的性能进行衡量。其中优秀>良好>中>差。但为了更好对性能进行比较，类似于1中的模型，我们同样地对进行量化。即按照优秀（9~10）、良好（6~8）、中（3~5）、差（1~2）来量化其对性能的影响值。再根据一些客观材料，对数据范围进一步缩减得到如下表：



但在实际的轮胎花纹走向中，大多数不会是完全的横向走向或完全的纵向走向，故为了更好设计轮胎性能更好的花纹，我们定义参数x，其中x表示横向花纹所占所有花纹块的面积比。

其次，对于大多数横向走向的花纹，实际上其并非完全的横向，而是与横向之间有一个夹角  $\theta$  ，所以，我们在设计花纹走向的时候引进这两个参数。

在引进这两个参数后，走向对轮胎性能的影响事实上已经确定了。也就是说，走向对轮胎性能的影响只与  $\mathbf{x}$  与  $\theta$  有关。

为了更好地计算这些影响值，我们定义三个矩阵A,B,C。A矩阵代表不同的花纹走向（横向、纵向）的影响值，B代表不同的沟槽比的影响值，C代表不同的沟深的影响值。即

$$
A = \left\{ \begin{array}{llllll}9 & 4 & 9 & 1 & 1 & 1\\ 1 & 10 & 5 & 8 & 1 & 9 \end{array} \right\} \tag{6-1}
$$

# 第七届数学中国数学建模网络挑战赛


$$
B = \left\{ \begin{array}{llllll}9 & 9 & 4 & 5 & 4 & 2\\ 8 & 5 & 5 & 7 & 8 & 7\\ 5 & 5 & 4 & 5 & 4 & 4\\ 9 & 9 & 9 & 1 & 2 & 1 \end{array} \right\} \tag{6-2}
$$

$$
C = \left\{ \begin{array}{llllll}10 & 1 & 2 & 4 & 1 & 2\\ 6 & 4 & 8 & 4 & 7 & 5\\ 8 & 7 & 10 & 4 & 8 & 5\\ 2 & 9 & 4 & 4 & 9 & 1 \end{array} \right\} \tag{6-3}
$$

对于矩阵A，为了计算出  $\theta$  和  $\mathbf{x}$  对于轮胎的性能的影响，我们将矩阵A进行分块得

$$
A = \left\{ \begin{array}{l}\alpha_{1}\\ \alpha_{2} \end{array} \right\} \tag{6-4}
$$

其中向量  $\alpha_{1}$  代表横向花纹对轮胎性能的影响，向量  $\alpha_{2}$  代表纵向花纹对轮胎性能的影响。在引进参数  $\mathbf{x}$  和  $\theta$  后，经过我们可得到矩阵D，且有

$$
D = \left\{ \begin{array}{l}\delta_{1}\\ \delta_{2} \end{array} \right\} = \left\{ \begin{array}{c}\alpha_{1}\times \cos \theta +\alpha_{2}\times \sin \theta \\ \alpha_{2} \end{array} \right\} \tag{6-5}
$$

$\delta_{1}$  为与横向倾斜角  $\theta$  后的对花纹轮胎的影响，  $\delta_{2}$  为纵向花纹对轮胎性能的影响。则花纹走向对轮胎的性能总的影响表示为：

$$
\alpha_{all} = x\delta_{1} + (1 - x)\delta_{2} = x\times (\alpha_{1}\times \cos \theta +\alpha_{2}\times \sin \theta) + (1 - x)\times \alpha_{2} \tag{6-6}
$$

6.2.2花纹子块间的组合模型

花纹走向、沟槽比、沟深是我们在设计花纹的时候最重要的三个要素，也是影响轮胎性能的三个重要的因素。在模型2中，已经明确地给出了这三个要素对于轮胎花纹性能影响的计算。而在实际设计花纹的过程中，我们需要满足特定的使用需求，即要满足一定的轮通过胎性能的需求，此时，即可通过一系列的约束条件来确定满足条件的花纹参数的组合。以下来具体列出求解步骤。

首先，我们来具体的列出决定花纹设计的四个参数，其分别是  $\mathbf{x},\theta$  ，以及b，c。其中x，  $\theta$  表示花纹的走向，b表示沟槽比，c表示沟深。而向量target  $\coloneqq [t_{1},t_{2},\dots ,t_{6}]$  表示在我们所设定的条件下的要求的最低的性能指标。则我们可列出一下的方程组：

# 第七届数学中国数学建模网络挑战赛


$$
\begin{array}{r l}&{\left\{\begin{array}{l l}{x\times(\alpha_{11}\times\cos\theta+\alpha_{21}\times\sin\theta)+(1-x)\times\alpha_{21}+y_{1}\times b_{11}+y_{2}\times b_{21}+y_{3}\times b_{31}+y_{4}\times b_{41}+z_{1}\times c_{11}}\\ {+z_{2}\times c_{21}+z_{3}\times c_{31}+z_{4}\times c_{41}>=t_{1}}\\ {x\times(\alpha_{12}\times\cos\theta+\alpha_{22}\times\sin\theta)+(1-x)\times\alpha_{22}+y_{1}\times b_{12}+y_{2}\times b_{22}+y_{3}\times b_{32}+y_{4}\times b_{42}+z_{1}\times c_{12}}\\ {+z_{2}\times c_{22}+z_{3}\times c_{32}+z_{4}\times c_{42}>=t_{2}}\\ {x\times(\alpha_{13}\times\cos\theta+\alpha_{23}\times\sin\theta)+(1-x)\times\alpha_{23}+y_{1}\times b_{13}+y_{2}\times b_{23}+y_{3}\times b_{33}+y_{4}\times b_{43}+z_{1}\times c_{13}}\\ {+z_{2}\times c_{23}+z_{3}\times c_{33}+z_{4}\times c_{43}>=t_{3}}\\ {x\times(\alpha_{14}\times\cos\theta+\alpha_{24}\times\sin\theta)+(1-x)\times\alpha_{24}+y_{1}\times b_{14}+y_{2}\times b_{24}+y_{3}\times b_{34}+y_{4}\times b_{44}+z_{1}\times c_{14}}\\ {+z_{2}\times c_{24}+z_{3}\times c_{34}+z_{4}\times c_{44}>=t_{4}}\\ {x\times(\alpha_{15}\times\cos\theta+\alpha_{25}\times\sin\theta)+(1-x)\times\alpha_{25}+y_{1}\times b_{15}+y_{2}\times b_{25}+y_{3}\times b_{35}+y_{4}\times b_{45}+z_{1}\times c_{15}}\\ {+z_{2}\times c_{25}+z_{3}\times c_{35}+z_{4}\times c_{45}>=t_{5}}\\ {x\times(\alpha_{16}\times\cos\theta+\alpha_{26}\times\sin\theta)+(1-x)\times\alpha_{26}+y_{1}\times b_{16}+y_{2}\times b_{26}+y_{3}\times b_{36}+y_{4}\times b_{46}+z_{1}\times c_{16}}\\ {+z_{2}\times c_{26}+z_{3}\times c_{36}+z_{4}\times c_{46}>=t_{6}}\end{array}\right.}\end{array}
$$

其中

$y_{1} + y_{2} + y_{3} + y_{4} = 1$  且  $y_{1},y_{2},y_{3},y_{4}$  为0- 1变量，分别表示沟槽比0.1,0.3，0.5，0.7； $z_{1} + z_{2} + z_{3} + z_{4} = 1$  且  $z_{1},z_{2},z_{3},z_{4}$  为0- 1变量，分别表示沟深2mm，5mm，8mm，11mm； $x\in [0,1],\theta \in [0,\mathrm{pi} / 2]$  ，即  $\mathbf{X}$  和  $\theta$  均是连续的

理论上通过这组约束条件，即可得到一组满足条件的  $x,\theta ,y_{i},z_{i}$  ，从而可以确定轮胎花纹的设计方案。但考虑到上述问题中六个不等式均为非线性规划，而且  $x,\theta$  均为连续值，无法给出所有的可能的结果，故此组约束条件的求解是十分困难的。因此我们考虑对该非线性规划进行简化处理。

事实上，工业上在设计轮胎花纹的时候，  $x,\theta$  的值不会随着每一组解的不同而进行机械设备上的大调整。故，我们考虑将  $x,\theta$  定格在某几个值之间，然后运用图论中的知识进行组合，在经过筛选来得到我们所需要的轮胎花纹的组合。具体实施的算法如下：

(1).设定  $x,\theta$  的向量  $x$  ，thet，由资料显示与实际生活经验可以得到以下的分类：即 $x = [0,0.2,0.4,0.6,0.8]$  ，thet  $= [\pi /12,\pi /6,\pi /4,\pi /3,5\pi /12]$  。在上述讨论中，也可到b，c的集合，令向量  $b = [0.1,0.3,0.5,0.7]$  ，  $c = [2\mathrm{mm},5\mathrm{mm},8\mathrm{mm},11\mathrm{mm}]$

(2).将向量  $x,thet,b,c$  的每个元素看成图G中的一个顶点，其中点集  $V_{x}$  每个点与点集 $V_{thet}$  中的每个点相连，点集  $V_{thet}$  与点集  $V_{b}$  中的每个点相连，点集  $V_{b}$  与点集  $V_{c}$  中的每个点相连。

(3).则上述讨论的问题即可转化为在图G中寻求一条连接点集  $V_{x}$  、  $V_{thet}$  、  $V_{b}$  、  $V_{c}$  最优路径。

(4).定义每个点的权值由其对应的轮胎性能向量来表示。

(5).由点集  $V_{x}$  有5个顶点，  $V_{thet}$  有5个顶点，  $V_{b}$  、  $V_{c}$  各有4个顶点，故其所有的组合的路径有400个，则我们可以通过MATLAB对所有的路径进行遍历，并计算出每条路径的各个轮胎性能的指标，并储存在矩阵com。具体的程序代码见附录essential.m，附录【2】中列举出了400种组合的前100组情况。

# 第七届数学中国数学建模网络挑战赛


(6). 通过对性能目标向量target与矩阵com进行比较，即可得到满足条件的所有可行路径。

# 6.3寻求最优路径模型的建立

通过所建立的模型，我们得到了在特定条件下满足条件需求的多组解的集合，但在实际的轮胎花纹设计的过程中，只会针对某种特定的情况设计出一种特定的花纹，因此，我们在设计花纹的过程中，需要对所得到的多种情况进行筛选，来寻求一个最优值出来。

对于筛选的过程，我们采用了TOPSIS（理想解法）法[10]。TOPSIS法是一种有效的多指标评价方法。这种方法通过构造评价问题的正理想解和负理想，及各项指标的最优解和最劣解，通过计算每个方案到理想方案的相对贴近度，即靠近正理想解和远离负理想解的程度，来对方案进行排序，从而选出最优方案[11]。

TOPSIS具体算法的步骤如下：

（1）用向量规范化的方法求得规范决策矩阵。在上个模型中，我们得到了所有可行解所构成的矩阵  $combinc = (a)_{ixj}$ ，对其进行规范化处理，得到规范化决策矩阵weight_norm  $= (\mathbf{b})_{ixj}$ ，其中

$$
b_{ij} = a_{ij} / \sqrt{\sum_{i = 1}^{m}a_{ij}^{2}}, \quad i = 1,2,\dots ,m; \quad j = 1,2,\dots ,n. \tag{6-7}
$$

（2）构成加权规范矩阵weight_norm  $= (c)_{ixj}$ 。对于权值矩阵的设定，我们是根据给定条件所得到对各个性能指标的目标向量  $target = (d)_{ixj}$  中  $d_{1j}$  的值来确定在第j个性能指标的权值  $w_{j}$ ，从而构成权重向量  $w = [\mathbf{w}_{1},\mathbf{w}_{2},\dots ,\mathbf{w}_{n}]$ ，即

$$
w_{j} = d_{ij} / \sum_{j = 1}^{n}d_{ij}, j = 1,2,\dots ,n \tag{6-8}
$$

$$
c_{ij} = w_{j}\times b_{ij}, i = 1,2,\dots ,m; j = 1,2,\dots ,n. \tag{6-9}
$$

（3）确定正理想解weight_norm*和负理想解weight_norm*。设正理想解weight_norm*的第j个属性值  $c_{j}^{*}$ ，负理想解weight_norm*第j属性值为  $c_{j}^{0}$ ，则

$$
\begin{array}{r}\mathbf{\Phi}_i = \left\{ \begin{array}{ll}\max_i c_{ij},\\ \min_i c_{ij}, \end{array} \right.\\ \mathbf{\Phi}_j = 1,2,\dots ,n;\\ \mathbf{\Phi}_i = \left\{ \begin{array}{ll}\max_i c_{ij},\\ \min_i c_{ij}, \end{array} \right.\\ \mathbf{\Phi}_j = 1,2,\dots ,n; \end{array} \tag{6-10}
$$

（5）计算各方案到正理想解与到负理想解的距离。备选方案  $d_{i}$  到正理想解的距离为

$$
s_{i}^{*} = \sqrt{\sum_{j}^{n}(\mathbf{c}_{ij} - \mathbf{c}_{j}^{*})^{2}}, i = 1,2,\dots ,m; \tag{6-12}
$$

备选方案  $d_{i}$  到负理想解的距离为

# 第七届数学中国数学建模网络挑战赛


$$
s_{i}^{0} = \sqrt{\sum_{j}^{n}(\mathbf{c}_{ij} - \mathbf{c}_{j}^{0})^{2}}, i = 1,2,\dots ,\mathbf{m}; \tag{6-13}
$$

（5）计算各方案的排队指标值（即综合评价指数），即

$$
f_{i}^{*} = s_{i}^{0} / \left(s_{i}^{0} + s_{i}^{*}\right), i = 1,2,\dots ,\mathbf{m}. \tag{6-14}
$$

（6）按  $f_{i}^{*}$  由大到小排列方案的优劣次序[12]

# 6.4 模型的求解

通过以上分析和假设，我们使用MATLAB软件，编写allt.m文件（见附录）对轿车、水泥地、安全性能三个影响因素（表6- 7）进行了计算分析。

表6-7  


得到轮胎性能要求指数  $F$

$$
F = (15 21 13.2 11.4 16.2 16.8),
$$

可见，考虑到表中的车辆类型、路面情况、使用需求时，对轮胎的防滑性能的要求相对要高一些。

通过，MATLAB软件编写essential.m（见附录）对所有花纹组合方案所能达到的轮胎性能与  $F$  进行比较，筛选出21组可行值。再用Topsis方法（相关Matlab程序：create_norm.m，topsis.m详见附录）对这些可行解进行处理排序，选出一项最优的可行解为：

$$
s o l u t i o n = \left(0.6\quad \frac{\pi}{3}\quad 0.3\quad 8\right), \tag{6-15}
$$

表6-8最优的花纹设计方案  


根据以上各量，使用AutoCAD软件对花纹进行简单设计得到以下图案：

# 第七届数学中国数学建模网络挑战赛


图7-1

我们从谷歌搜索，搜索到以下的两种轿车轮胎设计方案：

图7-2

通过对比图7- 1，可以看到，由我们的模型所确定的花纹设计方案能够基本的符合实际中工业设计的各个特点。对于工业轮胎设计中的细节，如花纹横块的大小排列、纵纹上的凹槽设计，由于模型的限制，并不能通过计算求得。

# 七．模型的优点和缺点

# 7.1模型的优点：

1. 较好的结合了车辆类型、路面状况、使用需求三个因素对花纹设计的影响程度，采用加权的方式放反映了实际设计中，不同因素的考虑权重；

# 第七届数学中国数学建模网络挑战赛


2. 对花纹设计的要求，考虑比较全面，将花纹分成若干子块，考虑每子块对轮胎性能的贡献，简化思路；3. 模型的可行解数若干，采用TOPSIS法确定最优解，体现了优化设计方案、采用最佳设计的原则。

# 7.2模型的缺点：

1、模型没有考虑花纹子块间的组合，互相造成的性能损失或性能增强的影响。

# 八．参考文献

八．参考文献[1]. Z周松波，左鸿，朱琨，工程车辆轮胎的选择与使用，建筑机械，58-60,2002(4)[2]. 李红波，野外作业中轮胎的选择与使用，物探装备，第15卷第3期，176-216,2005(9)[3]. 太平洋汽车网重庆分会版主，升级轮胎根据路况选择轮胎，http://bbs.pcauto.com.cn/topic-787725. html，2014-5-17[4]. 汽车画刊网，表面的纹章浅析轮胎花纹的奥秘，http://www.autobild.com.cn/test/201309-817068- all.html，2014-5-18[5]. 臧孟炎，朱林培，应卓凡，3- D轮胎模型划水仿真研究，科学技术与工程，第9卷第11期，2999-3002,2009(6)[6]. 李松龄，裴玉龙，路面附着性能影响因素分析及其改善对策的研究，公路，11期，126-130,2007(11)[7]. 陈振艺，不规则横向细沟槽胎面花纹噪声研究，轮胎工业，29卷283-287,2009年[8]. 360汽车网，轮胎选购，认黄路况与季节，http://www.360qc.com/news/NewGuide/201105/4512. html，2014-5-17[9]. 刘哲义，对影响轮胎与路面间附着性能因素的分析，公路，第6期，48-51，2000(6)[10]. 李磊，金菊良，朱永楠，TOPSIS方法应用中若干问题的探讨，水电能源科学，第30卷第3期，51-54,2012(3)[11]. 张吉军，樊玉英，权重为区间数的多指标决策问题的逼近理想点法，系统工程与电子技术，第24卷第11期，76-77,2002[12]. 司守奎，孙玺菁，数学建模算法与应用，国防工业出版社，345-351，北京，2014(2)

# 九.附录

# 1、MATLAB程序

（1）计算影响因素的不同组合的总体影响程度，根据指标target筛选出指标以上的组合。function[routes,combine,all]  $=$  essential(theta,x,a,b,c,target)%Traverse all the combinations of elements in a, b, c, caculate weight of routes connectedto a, b, c combin  $\equiv$  [];routes  $\equiv$  [];routes  $\equiv$  [];combinc  $\equiv$  [];all  $\equiv$  [];for  $\mathrm{i} = 1:5$  for  $\mathrm{j = 1:6}$  extent(i,j)=a(1,j)\*cos(theta(i))+a(2,j)\*sin(theta(i));endendfor  $\mathrm{i} = 1:6$  extent(6,i)=a(2,i);endfor  $\mathrm{k} = 1:5$

# 第七届数学中国数学建模网络挑战赛


for  $\mathrm{i} = 1:5$  for  $\mathrm{j} = 1:6$  combin1  $\scriptstyle \mathrm{(i,j) = x(k)^{*}extent(i,j) + (1 - x(k))^{*}extent(6,j);}$  end end combin  $\equiv$  [combin;combin1]; end for  $\mathrm{i} = 1:25$  for  $\mathrm{j} = 1:4$  combinb(j,:)=combin(i,:)+b(j,:); for  $\mathrm{k} = 1:4$  combinc1(k,:)=combinb(j,:)+c(k,:); flag  $\scriptstyle = 0$  for  $\mathrm{m} = 1:6$  if combinc1(k,m)- target(m)<0 flag  $\scriptstyle = 1$  continue; end end if flag  $\scriptstyle = =0$  routes1=[i,j,k]; routes  $\equiv$  [routes;routes1]; combinc  $\equiv$  [combinc;combinc1(k,:)]; end end all=[all;combinc1]; end end

（2）Topsis 法：将原始的组合信息combinc 转换成规范矩阵function norm_matrix  $\equiv$  create_norm(combinc) $\%$  combinc - Oringinal Data[m,n]=size(combinc);for  $\mathrm{j = 1:n}$  norm_matrix(:,j)=combinc(:,j)/norm(combinc(:,j));end

（3）将整合路径分布成原始的组合路径，即花纹设计方案的组合function rou=translatorou(routes,theta,x,bb,cc)[m,n]=size(routes);rou  $\equiv$  [];rou_rest  $\equiv$  [];for  $\mathrm{i} = 1:\mathrm{m}$  if routes(i,1)<5o=routes(i,1);elseo=fix(routes(i,1)/5);

# 第七届数学中国数学建模网络挑战赛


end p=mod(routes(i,1),5); if  $\mathrm{p} = =0$ $\mathrm{p} = 5$  end rou1=[theta(o),x(p)]; rou=[rou;rou1]; end for  $\mathrm{i} = 1:\mathrm{m}$  rou2=[bb(routes(i,2)),cc(routes(i,3))]; rou_rest=[rou_rest;rou2]; end rou=[rou,rou_rest];

（4）Topsis法：取最优可行解 function [sf,index]=topsist(weightednorm) [m,n]=size(weightednorm); c_positive  $\equiv$  max(weightednorm); c_negetiv  $\coloneqq$  min(weightednorm); for  $\mathrm{i} = 1:\mathrm{m}$  s_positive(i)=norm(weightednorm(i,:)-c_positive); s_negetiv  $\coloneqq$  norm(weightednorm(i,:)-c_negetive); end figure  $\equiv$  s_negetive./(s_negetive+s_positive); [sf,index]=sort(figure,'descend');

（5）根据车辆类型、道路状况、使用需求组合加权得出指标矩阵 function [all_target,target]=allt(car,conditions,needs) car=0.4\*car; conditions  $= 0.2*$  conditions; needs  $= 0.4*$  needs; part=[];all_target  $\equiv$  [];index0=[];index  $\equiv$  []; for  $\mathrm{i} = 1:5$  for  $\mathrm{j} = 1:4$  part1(j,:)=car(i,:)+conditions(j,:); index1(j,:)=[i,j]; end part=[part,part1]; index0=[index0;index1]; end for  $\mathrm{i} = 1:20$  for  $\mathrm{j} = 1:4$  part2(j,:)=part(i,:)+needs(j,:); index2(j,:)=[index0(i,:),j]; end all_target=[all_target,part2];

# 第七届数学中国数学建模网络挑战赛


index=[index;index2]; end target=3*all_target; all_target=[target,index];

2、部分数据

（1）、花纹设计子块的组合对轮胎性能影响的前一百种组合状态


# 第七届数学中国数学建模网络挑战赛



# 第七届数学中国数学建模网络挑战赛



（2）对每一指标  $F$  的最优可行解（前40组）


# 第七届数学中国数学建模网络挑战赛
