# A Survey: Deep Learning-based Time Series Forecasting from 2014 to 2024

>     	With the advancement of deep learning algorithms and the growing availability of computational power, deep learning-based forecasting methods have gained significant importance in the domain of time series forecasting. In the past decade, there has been a rapid rise in time series forecasting approaches. This paper comprehensively reviews the advancements in deep learning-based forecasting models spanning 2014 to 2024. 
>     We provide a comprehensive examination of the capabilities of these models in capturing correlations among time steps and time series variables. Additionally, we explore methods to enhance the efficiency of long-term time series forecasting and summarize the diverse loss functions employed in these models. Moreover, this study systematically evaluates the effectiveness of these approaches in both univariate and multivariate time series forecasting tasks across diverse domains. We comprehensively discuss the strengths and limitations of various algorithms from multiple perspectives, analyze their capacity to capture different types of time series information, including trend and season patterns, and compare methods for enhancing the computational efficiency of these models. Finally, we summarize the experimental results and discuss the future directions in time series forecasting.



# Install dependecies

Install the required packages

```bash
pip install -r requirements.txt
```



# Data Preparation

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



# Experimental setup

The length of the historical input sequence is maintained at 96(or 36 for the illness dataset), whereas the length of the sequence to be predicted is selected from a range of values, i.e., $48,96,336$ ($24,36,48$ for the illness dataset). Note that the input length is fixed to be 96 for all methods for a fair comparison. The evaluation is based on the mean squared error (MSE) and mean absolute error (MAE) metrics



# Get Started

We have provided all the experimental scripts for the benchmarks in the `./scripts` folder, which cover all the benchmarking experiments. To reproduce the results, you can run the following shell code.

```bash
   ./scripts/train.sh
```



# Citation

 Xiaobao Song, Hao Wang*, Liwei Deng, Yaoan Zhang, Yuxin He and WenmingCao *“Deep Learning-based Time Series Forecasting from 2014 to 2024*”,  submitted to Artificial Intelligence Review (under review)

