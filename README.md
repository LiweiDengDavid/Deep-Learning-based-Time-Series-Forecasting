# A Survey: Deep Learning-based Time Series Forecasting 

A pytorch implementation for the paper: ' *Deep Learning-based Time Series Forecasting*'  Xiaobao Song, Liwei Deng,Hao Wang, Yaoan Zhang, Yuxin He and Wenming Cao (*Correspondence)

# Introduction

![](/image/process.jpg)

# Model Statics 

![Model](/image/Model.png)

# Dataset Statics

![Dataset](/image/Dataset.png)

# Get Started

<span id='all_catelogue'/>

### Table of Contents:

- <a href='#Install dependecies'>1. Install dependecies</a>
- <a href='#Data Preparation'>2. Data Preparation </a>
- <a href='#Run Experiment'>3. Run Experiment</a>

<span id='Install dependecies'/>

## Install dependecies  <a href='#all_catelogue'>[Back to Top]</a>

Install the required packages

```bash
pip install -r requirements.txt
```

<span id='Data Preparation'/>

# Data Preparation<a href='#all_catelogue'>[Back to Top]</a>

We follow the same setting as previous work. The datasets for all the six benchmarks can be obtained from [[Autoformer](https://github.com/thuml/Autoformer)]. The datasets are placed in the 'all_six_datasets' folder of our project. The tree structure of the files are as follows:

```
Dateformer\datasets
├─electricity
│
├─ETT-small
│
├─exchange_rate
│
├─illness
│
└─traffic
```

<span id='Run Experiment'/>

# Run Experiment<a href='#all_catelogue'>[Back to Top]</a>

We have provided all the experimental scripts for the benchmarks in the `./scripts` folder, which cover all the benchmarking experiments. To reproduce the results, you can run the following shell code.

```bash
 ./scripts/ETTh1.sh
 ./scripts/ETTh2.sh
 ./scripts/ETTm1.sh
 ./scripts/ETTm2.sh
 ./scripts/exchange.sh
 ./scripts/illness.sh
 ./scripts/traffic.sh
```



## Contact

For any questions or feedback, feel free to contact [Xiaobao Song](2840329517@qq.com) or [Liwei Deng](liweidengdavid@gmail.com).

# Citation

If you find this code useful in your research or applications, please kindly cite: 

Xiaobao Song, Liwei Deng,Hao Wang*, Yaoan Zhang, Yuxin He and Wenming Cao *“Deep Learning-based Time Series Forecasting*”,  accepted by **Artificial Intelligence Review**



# Acknowledgments

We express our gratitude to the following members for their contributions to the project, completed under the guidance of Professor [Hao Wang](haowang@szu.edu.cn):

[Xiaobao Song](2840329517@qq.com)， [Liwei Deng](liweidengdavid@gmail.com)，[Yaoan Zhang](2291149420@qq.com)，[Junhao Tan](827092078@qq.com)，[Hongbo Qiu](2023280567@email.szu.edu.cn)，[Xinhe Niu](Jack1299952745@gmail.com)

