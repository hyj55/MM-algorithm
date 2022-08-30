# MM algorithm
## Table of Contents
- [General Informations](#general_info)
- [Installation](#install)
- [Usage](#useage)
    - [Class MM_algorithm](#mm_alg)
    - [Class Data_generation](#data_generation)
    - [Example](#exp)


## General Informations <a name="general_info"></a>
A package for MM algorithm for solving MLE of latent scores of each player in pairwise comprison models.\
Note that here the pairwise compirisons are measured on an ordinal scale. The package contains algorithms for models supporting the ordinal scale of levels J=2,3,4,5.\
Besides the algorithms, the package also provides corresponding data generations for models of levels J=3,4,5 for simulation use.
For function informations of both classes together with code examples, please find in the **Usage**.

## Installation <a name="install"></a>


## Usage <a name="useage"></a>
Function informations including function discriptions, inputs & outputs are listed in **Class MM_algorithm** \& **Class Data_generation**, detailed usage with code examples are provided in **Example**.

### Class MM_algorithm <a name="mm_alg"></a>
The list of functions with their corresponding inputs & outputs in ***Class MM_algorithm*** for consultation.
#### functions
|function name|discription|inputs|outputs|
|-|------------------|----------|--------|
|mmAlgorithm_bt(iteration=1000,error=1e-5)|MM algorithm for Bradely-Terry Model, only for J=2 cases|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> **class input**:<br>-*size*<br>-*graphs* for J=2|estimation(for latent scores gamma)|
|Davidson(iteration=1000,error=1e-9)|MM algorithm for Davidson Model, only for J=3 cases, theta is to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> **class input**:<br>-*size*<br>-*graphs* for J=3|estimation, estimation_theta|
|Davidson_given_theta(iteration=1000,error=1e-9,theta=1)|MM algorithm for Davidson Model, only for J=3 cases, theta is given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*,<br>-*theta*(another parameter in model for allowing existance of ties) <br>**class input**:<br>-*size*<br>-*graphs* for J=3|estimation|
|Rao_Kupper(iteration=1000,error=1e-9)|MM algorithm for Rao-Kupper Model, only for J=3 cases, theta is to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> **class input**:<br>-*size*<br>-*graphs* for J=3|estimation, estimation_theta|
|Rao_Kupper_given_theta(iteration=1000,error=1e-9,theta=1.5)|MM algorithm for Rao-Kupper Model, only for J=3 cases, theta is given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*,<br>-*theta*<br>**class input**:<br>-*size*<br>-*graphs* for J=3|estimation|
|clm(iteration=50000,error=1e-6)|MM algorithm for Cumulative Link Model, only for J=4 cases, theta is to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> **class input**:<br>-*size*<br>-*graphs* for J=4|estimation, estimation_theta|
|clm_given_theta(iteration=5000,error=1e-6,theta=2)|MM algorithm for Cumulative Link Model, only for J=4 cases, theta is given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> -*theta* <br>**class input**:<br>-*size*<br>-*graphs* for J=4|estimation|
|aclm(iteration=5000,error=1e-6)|MM algorithm for Adjacent Categories Logit Model, only for J=4 cases, theta is to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> **class input**:<br>-*size*<br>-*graphs* for J=4|estimation, estimation_theta|
|aclm_given_theta(iteration=5000,error=1e-6,theta=2)|MM algorithm for Adjacent Categories Logit Model, only for J=4 cases, theta is given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> -*theta* <br>**class input**:<br>-*size*<br>-*graphs* for J=4|estimation|
|clm_5(iteration=50000,error=1e-6)|MM algorithm for Cumulative Link Model, only for J=5 cases, thetas are to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> **class input**:<br>-*size*<br>-*graphs* for J=5|estimation, estimation_theta_1, estimation_theta_2|
|clm_5_given_theta(iteration=50000,error=1e-6,theta_1=3,theta_2=3)|MM algorithm for Cumulative Link Model, only for J=5 cases, thetas are given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> -*theta_1*, *theta_2* <br>**class input**:<br>-*size*<br>-*graphs* for J=5|estimation|
|aclm_5(iteration=50000,error=1e-6)|MM algorithm for Adjacent Categories Logit Model, only for J=5 cases, thetas are to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> **class input**:<br>-*size*<br>-*graphs* for J=5|estimation, estimation_theta_1, estimation_theta_2|
|aclm_5_given_theta(iteration=50000,error=1e-6,theta_1=3,theta_2=3)|MM algorithm for Adjacent Categories Logit Model, only for J=5 cases, thetas are given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*<br> -*theta_1*, *theta_2* <br>**class input**:<br>-*size*<br>-*graphs* for J=5|estimation|
|check_assumption_3_no_tie()|check Assumption 3, <br>would remove some players in the way of truncating their comparison records in the comparison graphs (\**graphs) for ensuring convergence of the algorithm, <br>only for J=2 cases|no function input<br>**class input**:<br>-*size*<br>-*graphs* for J=2|index|
|check_assumption_3|check Assumption 3, <br>would remove some players in the way of truncating their comparison records in the comparison graphs (\**graphs) for ensuring convergence of the algorithm, <br> only for J=3 cases|no function input<br>**class input**:<br>-*size*<br>-*graphs* for J=3|index|
|check_assumption_1|check Assumption 1, <br>would remove some players in the way of truncating their comparison records in the comparison graphs (\**graphs) for ensuring convergence of the algorithm, <br> only for J=3 cases|no function input<br>**class input**:<br>-*size*<br>-*graphs* for J=3|index|

Detailed discriptions for inputs & outputs of class and its functions are listed below:
#### Inputs of ***class MMAlgorithm***
|name|discription|
|----|-----------|
|size|***int(>0)*** <br> The number of players|
|**graphs|***ndarray of shape (size,size)***, *the input format must be:* <br>``` # when J = 5: ```<br> ```graph_1 = graph_1, graph_2 = graph_2, graph_3 = graph_3, graph_4 = graph_4, graph_5 = graph_5; ```<br> ```# when J = 4: ```<br> ```graph_1 = graph_1, graph_2 = graph_2, graph_3 = graph_3, graph_4 = graph_4;``` <br> ```# when J = 3: ```<br>``` win_graph = win_graph, tie_graph = tie_graph;``` <br> ```# when J = 2: ```<br> ```win_graph = win_graph``` <br> The pairwise comparison graphs, record the results of comparisons for each level.<br> i.e. The *ij_th* term of *graph_4* records the number of times the comparisons between players *i* and *j* are measured to be 4.|
#### Inputs of functions
|name|discription|
|----|-----------|
|iteration|***int (>0)***,<br> the maximum number of iterations allowed|
|error|***float***,<br>the $l_{\infty}$ norm of the difference between estimated latent scores $\gamma$s from the last two iterations, <br> i.e. $\lVert\gamma^{k-1}-\gamma^{k}\rVert_{\infty}$, where *k* is the last iteration.|
|theta|***float (>=1)***,<br> another parameter $\theta$ in model of level J=3,4; related to the cutpoints for level measurements.|
|theta_1|***float (1<=theta_1<=theta_2)***,<br> a parameter $\theta_1$ except for $\gamma$ in model of level J=5; related to the cutpoints for level measurements.|
|theta_2|***float (1<=theta_1<=theta_2)***,<br> another parameter $\theta_2$ except for $\gamma$ in model of level J=5; also related to the cutpoints for level measurements.|
#### Outputs of functions
|name|discription|
|----|-----------|
|estimation|***ndarray of shape (size,)***<br> The estimated value of latent scores of players $\gamma$|
|estimation_theta|***float***,<br>The estimated value of $\theta$|
|estimation_theta_1|***float***,<br>The estimated value of $\theta_1$|
|estimation_theta_2|***float***,<br>The estimated value of $\theta_2$|
|index|***ndarray of shape (size\*,),*** *size\** is the number of players after the truncation.<br> The index of the remaining players after the truncation.|

### Class Data_generation <a name="data_generation"></a>
The data generation of the inputs required by the above algorithms are provided for simulation/testing.
Below find the lists of functions with their corresponding inputs & outputs in ***Class Data_generation*** for consultation.
#### functions
|function name|discription|inputs|outputs|
|-|---------------------           |--|--|
|data_generation_rao|Generates data for Rao-Kupper Model, only for J=3|no function input.<br> ***class inputs***:<br> size, num_game, dynamic_range, sparsity, <br>\*\*theta (J<3)|Players, win_graph, tie_graph, gamma|
|data_generation_davi|Generates data for Davidson Model, only for J=3|no function input.<br> ***class inputs***:<br> size, num_game, dynamic_range, sparsity, <br>\*\*theta (J=3,4)|Players, win_graph, tie_graph, gamma|
|generate_four_data_clm|Generates data for Cumulative Link Model, only for J=4|no function input.<br> ***class inputs***:<br> size, num_game, dynamic_range, sparsity,<br> \*\*theta (J=3,4)|graph_4, graph_3, graph_2, graph_1, gamma|
|generate_four_data_aclm|Generates data for Adjacent Categories Logit Model, only for J=4|no function input.<br> ***class inputs***:<br> size, num_game, dynamic_range, sparsity, <br>\*\*theta (J=3,4)|graph_4, graph_3, graph_2, graph_1, gamma|
|generate_five_data_clm|Generates data for Cumulative Link Model, only for J=5|no function input.<br> ***class inputs***:<br> size, num_game, dynamic_range, sparsity, <br>\*\*theta (J=5)|graph_5, graph_4, graph_3, graph_2, graph_1, gamma|
|generate_five_data_aclm|Generates data for Adjacent Categories Logit Model, only for J=5|no function input.<br> ***class inputs***:<br> size, num_game, dynamic_range, sparsity, <br>\*\*theta (J=5)|graph_5, graph_4, graph_3, graph_2, graph_1, gamma|

#### Inputs of ***Class Data_generation***
|name|discription|
|-|-------          |
|size|***int (>0)***,<br> The number of players|
|num_game|***int (>0)***,<br> The maximum number of games between each pair of players|
|dynamic_range|***float (>0)***, <br> related to the range of latent score $\gamma$|
|sparsity|***float (\[0,1\])***, <br> related to the frequency of the occurence of a game between each pair of players|
|\*\*theta|***float (>1)*** *the input format must be:* <br> ```# when J=5:```<br>```theta_1 = theta_1, theta_2 = theta_2```<br>```# when J=4,3:```<br>```theta = theta```<br>```# when J<3:```<br>```theta=0```|

#### Outputs of functions
|name|discription|
|----|-------    |
|Players|***ndarray of shape (size,)***,<br> an array of the form $\[0,1,2,...,n\]$, where n is the number of players.|
|win_graph|***ndarray of shape (size,size)***, <br> The pairwise comparison graph, only record the wining results of comparisons.<br> i.e. The *ij_th* term of the graph records the number of times when player *i* beats *j*.|
|tie_graph|***ndarray of shape (size,size)***, <br>The pairwise comparison graphs, only record the tie results of comparisons.<br> i.e. The *ij_th* term of the graph records the number of times when players *i* and *j* are tied.|
|graph_n|***ndarray of shape (size,size)***, <br> The pairwise comparison graph, record the results of comparisons for each level.<br> i.e. The *ij_th* term of *graph_n* records the number of times the comparisons between players *i* and *j* are measured to be n.|
|gamma|***ndarray of shape (size,)***, <br> the latent scores of the players, i.e., gammma\[n\] is the score of n+1_th player.|
***Remark:***\
*The comparison graphs are generated according to the corresponding gamma and model.*\

### Example <a name="exp"></a>






