'''
Author: feng_chloe
Date: 2022-06-10 15:54:54
Description: python code python3.8+vscode+jupyter
FilePath: \undefinedc:\Users\20952\Desktop\课程设计\下午考试命题与数据源\task.py
'''
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# 任务1. （50%）
# 沪深300成分股是按照一定优选标准，对上市公司进行“移出”或“移入”的动态操作，调整频率为每半年一次。
# “任务1数据源”显示沪深300成分股上市公司2012-2015年列表(每半年一张数据页(worksheet))及其月收益率。
# 请编写Python或Matlab程序，完成以下数据分析任务：
# 1. 从数据源中提取自2012年下半年至2015年下半年，每次成分股调整被“移出”和“移入”上市公司的列表，并写入沪深300成分股异动记录表，每半年一页。
task1_income=[]
task1_company=[]
for i in range(8):
    task1_income.append(pd.read_excel('任务1数据源.xlsx',sheet_name=i,header=None).iloc[1:301,3:9])
    task1_company.append(pd.read_excel('任务1数据源.xlsx',sheet_name=i)['证券简称'].iloc[:300])

# %%
writer=pd.ExcelWriter('沪深300成分股异动记录表.xlsx')
company_active=[]
for i in range(7):
    company_active_temp={}
    for j in task1_company[i+1].tolist():
        if j not in task1_company[i].tolist():
            company_active_temp[j]='移入'
    for k in task1_company[i].tolist():
        if k not in task1_company[i+1].tolist():
            company_active_temp[k]='移出'
    key_c=[]
    value_c=[]
    for key,value in company_active_temp.items():
        key_c.append(key)
        value_c.append(value)
    company_active_i=pd.DataFrame([key_c,value_c]).T
    company_active_i.columns=['证券简称','异动情况']
    company_active_i.to_excel(writer,sheet_name='sheet'+str(i+1))
    company_active.append(company_active_temp)
writer.save()


# %%
# 第二部分
# “任务23数据源”为“五粮液”、“平安银行”、“三一重工”三家上市公司2006-2015年每月底的收益率数据（%），
# 任务2. （30%）
# 检验以上三个公司的月收益率是否各自符合正态分布？(检验方法可以任选kstest或lillietest)。
# 进行正态检验 选择K-S检验
writer=pd.ExcelWriter('相关系数矩阵.xlsx')
task2_income=pd.read_excel('任务23数据源.xlsx').iloc[:,1:]
from scipy.stats import kstest
statistic=[]
pvalue=[]
result=[]
for i in task2_income.columns:
   print(str(i)+'的正态检验结果为：')
   print(kstest(task2_income[i],cdf = "norm"))
   statistic.append(kstest(task2_income[i],cdf = "norm")[0])
   pvalue.append(kstest(task2_income[i],cdf = "norm")[1])
   if kstest(task2_income[i],cdf = "norm")[1]<0.05:
      result.append('不符合正态分布')
   else:
      result.append('符合正态分布')
task2_ks=pd.DataFrame([statistic,pvalue,result]).T
task2_ks.columns=['statistic','pvalue','检验结果']
task2_ks.index=['五粮液','平安银行','三一重工']
task2_ks.to_excel(writer,sheet_name='正态检验结果')

# %%
# 任务3. （20%）
# 计算以上三家公司之间的收益率之间的皮尔逊相关系数，写入相关系数矩阵excel表格。
# 3.皮尔逊相关系数
print('计算皮尔逊相关系数结果为：'+str(task2_income.corr(method='pearson')))
pd.DataFrame(task2_income.corr(method='pearson')).to_excel(writer,sheet_name='皮尔逊相关系数')
writer.save()


