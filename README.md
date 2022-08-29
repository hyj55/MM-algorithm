# MM algorithm
## Table of Contents
- [General Informations](#general_info)
- [Installation](#install)
- [Usage](#useage)
    - [MM_algorithm](#mm_alg)
    - [Data_generation](data_generation)


## General Informations <a name="general_info"></a>
A package for MM algorithm for solving MLE of latent scores of each player in pairwise comprison models.\
Note that here the pairwise compirisons are measured on an ordinal scale. The package contains algorithms for models supporting the ordinal scale of levels J=2,3,4,5.\
Besides the algorithms, the package also provides corresponding data generations for simulation use.

## Installation <a name="install"></a>


## Usage <a name="useage"></a>
The package contains two parts as mentioned above, MM_algorithm and Data_generation.

### Class MM_algorithm <a name="mm_alg"></a>
The list of functions with their corresponding inputs & outputs in ***Class MM_algorithm*** for consultation.
#### functions
|function name|discription|inputs|outputs|
|-------------|-----------|------|-------|
|mmAlgorithm_bt(iteration=1000,error=1e-5)|MM algorithm for Bradely-Terry Model, only for J=2 cases|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> **class input**:<br>-*size*<br>-*graphs* for J=2|estimation(for latent scores gamma)|
|Davidson(iteration=1000,error=1e-9)|MM algorithm for Davidson Model, only for J=3 cases, theta is to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> **class input**:<br>-*size*<br>-*graphs* for J=3|estimation, estimation_theta|
|Davidson_given_theta(iteration=1000,error=1e-9,theta=1)|MM algorithm for Davidson Model, only for J=3 cases, theta is given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error),<br>-*theta*(another parameter in model for allowing existance of ties) <br>**class input**:<br>-*size*<br>-*graphs* for J=3|estimation|
|Rao_Kupper(iteration=1000,error=1e-9)|MM algorithm for Rao-Kupper Model, only for J=3 cases, theta is to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> **class input**:<br>-*size*<br>-*graphs* for J=3|estimation, estimation_theta|
|Rao_Kupper_given_theta(self,iteration=1000,error=1e-9,theta=1.5)(iteration=1000,error=1e-9,theta=1)|MM algorithm for Rao-Kupper Model, only for J=3 cases, theta is given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error),<br>-*theta*<br>**class input**:<br>-*size*<br>-*graphs* for J=3|estimation|
|clm(iteration=50000,error=1e-6)|MM algorithm for Cumulative Link Model, only for J=4 cases, theta is to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> **class input**:<br>-*size*<br>-*graphs* for J=4|estimation, estimation_theta|
|clm_given_theta(iteration=5000,error=1e-6,theta=2)|MM algorithm for Cumulative Link Model, only for J=4 cases, theta is given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> -*theta* <br>**class input**:<br>-*size*<br>-*graphs* for J=4|estimation|
|aclm(iteration=5000,error=1e-6)|MM algorithm for Adjacent Categories Logit Model, only for J=4 cases, theta is to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> **class input**:<br>-*size*<br>-*graphs* for J=4|estimation, estimation_theta|
|aclm_given_theta(iteration=5000,error=1e-6,theta=2)|MM algorithm for Adjacent Categories Logit Model, only for J=4 cases, theta is given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> -*theta* <br>**class input**:<br>-*size*<br>-*graphs* for J=4|estimation|
|clm_5(iteration=50000,error=1e-6)|MM algorithm for Cumulative Link Model, only for J=5 cases, thetas are to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> **class input**:<br>-*size*<br>-*graphs* for J=5|estimation, estimation_theta_1, estimation_theta_2|
|clm_5_given_theta(iteration=50000,error=1e-6,theta_1=3,theta_2=3)|MM algorithm for Cumulative Link Model, only for J=5 cases, thetas are given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> -*theta_1*, *theta_2* <br>**class input**:<br>-*size*<br>-*graphs* for J=5|estimation|
|aclm_5(iteration=50000,error=1e-6)|MM algorithm for Adjacent Categories Logit Model, only for J=5 cases, thetas are to be estimated|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> **class input**:<br>-*size*<br>-*graphs* for J=5|estimation, estimation_theta_1, estimation_theta_2|
|aclm_5_given_theta(self,iteration=50000,error=1e-6,theta_1=3,theta_2=3)|MM algorithm for Adjacent Categories Logit Model, only for J=5 cases, thetas are given|**function input**: <br>-*iteration*(num_iteration),<br> -*error*(estimation error)<br> -*theta_1*, *theta_2* <br>**class input**:<br>-*size*<br>-*graphs* for J=5|estimation|
|check_assumption_3_no_tie()|check Assumption 3, only for J=2 cases|no function input<br>**class input**:<br>-*size*<br>-*graphs* for J=2|index|
|check_assumption_3|check Assumption 3, only for J=3 cases|no function input<br>**class input**:<br>-*size*<br>-*graphs* for J=3|index|
|check_assumption_1|check Assumption 1, only for J=3 cases|no function input<br>**class input**:<br>-*size*<br>-*graphs* for J=3|index|

Detailed discriptions for inputs & outputs of class and its functions are listed below:
#### Inputs of ***class MMAlgorithm***
|name|discription|
|----|-----------|
|size|***int(>0)*** <br> The number of players|
|**graphs|***ndarray of shape (size,size)***, *the input format must be:* <br>``` # when J = 5: ```<br> ```graph_1 = graph_1, graph_2 = graph_2, graph_3 = graph_3, graph_4 = graph_4, graph_5 = graph_5; ```<br> ```# when J = 4: ```<br> ```graph_1 = graph_1, graph_2 = graph_2, graph_3 = graph_3, graph_4 = graph_4;``` <br> ```# when J = 3: ```<br>``` win_graph = win_graph, tie_graph = tie_graph;``` <br> ```# when J = 2: ```<br> ```win_graph = win_graph``` <br> The pairwise comparison graphs, record the results of comparisons for each level.<br> i.e. The *ij_th* term of *graph_4* records the number of times the comparisons between players *i* and *j* are measured to be 4.|
#### Inputs of functions
|name|discription|
|----|-----------|
|||
#### Outputs of functions
|name|discription|
|----|-----------|
|estimation|***ndarray of shape (size,)***<br> The estimated value of latent scores of players|
|estimation_theta||
|estimation_theta_1||
|estimation_theta_2||
|index||

### Class Data_generation <a name="data_generation"></a>
The lists of functions with their corresponding inputs & outputs in ***Class Data_generation*** for consultation.

