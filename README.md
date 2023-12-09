# graph_embedding_link_scheduling

This repository contains the entire code for our TWC work "Graph Embedding based Wireless Link Scheduling with Few Training Samples", available at https://ieeexplore.ieee.org/document/9285223.

For any reproduce, further research or development, please kindly cite our TWC Journal paper:

M. Lee, G. Yu, and G. Y. Li, "Graph embedding based wireless link scheduling with few training samples," IEEE Trans. Wireless
Commun., vol. 20, no. 4, pp. 2282-2294, Apr. 2021.


# How to use?

## Requirements

The following versions have been tested. But newer versions should also be fine. 

- rdkit : [Q3 2017 Release](https://github.com/rdkit/rdkit/releases/tag/Release_2017_09_1, Release_2017_09_2)
- boost : Boost 1.61.0, 1.65.1

## Setup

Go to "s2v_lib". 

Build the c++ backend of s2v_lib and you are all set.

```
cd s2v_lib
make -j4  
```

Note: We utilize existing open-source code for the structure2vec architecture (https://github.com/Hanjun-Dai/pytorch_structure2vec/tree/master/s2v_lib) and add the batch nomarlization function in it.


## Data Prepareation

Go to  "FPLinQ".

Run "generate_main.m". The output data is saved in "/mat".

Copy the output data into  "D2D_qua/mat".

## Run

Go to "D2D_qua".

Run the main program.

```
./run.sh
```
You can also use the following command.
```
python main.py
```
