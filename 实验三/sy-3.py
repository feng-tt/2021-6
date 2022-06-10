'''
Author: feng_chloe
Date: 2022-06-09 22:34:53
Description: sy3 code python3.8+vscode+jupyter notebook
FilePath: \undefinedc:\Users\20952\Desktop\课程设计\实验三\sy-3.py
'''
# %%
# 对提供的多变量数据进行分析，从众多变量中选择满足以下条件的有效预测因子：
# 1、单因子分组序数与对应分组的收益率成强相关，正相关时相关系数大于 0.6，负相关时相关系数小于-0.6；
# 2、单因子优势组合的超额收益率在 95%的置信水平下显著为正；
# 3、单因子优势组合取得超额收益的频率需大于等于 60%（市场基准为沪深 300 指数）。

# %%
# 导包
# pip install pandas,numpy,scipy
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# %%
# 0.读取收益率
income=[]
for i in range(20):
    income.append(pd.read_excel("沪深300成分股票收益率.xlsx",sheet_name=i,header=None).iloc[1:301,4:10])

# 0.读取沪深300指数
CSI_Index=pd.read_excel('等比例投资组合收益率(1).xlsx',sheet_name=6,skiprows=17,header=None).iloc[:,3:123]
CSI_Index.set_axis(list(range(120)),axis='columns',inplace=True)

# %%
# 1.排序 升序 自下往上从低到高
# 排序函数rank 输入：因子A:三维list（20*300*6），待排序income三维list（20*300*6） 
# 输出：排序整合后的A_s_i三维list（20*300*6），与排序整合后的income三维list（20*300*6）
def rank(A,income):
    income_s=[]
    A_s=[]
    for i in range(20):
        income_temp=[]
        A_temp=[]
        income_temp=pd.Series(income_temp)
        A_temp=pd.Series(A_temp)
        for j in range(6):
            zipped=zip(np.array(income[i])[:,j].tolist(),np.array(A[i])[:,j].tolist())
            sorted_zipped=sorted(zipped,key=lambda x:(x[1],x[0]))
            result=zip(*sorted_zipped)
            sorted_income_temp,sorted_A_temp=[list(x) for x in result]

            income_temp=pd.concat([income_temp,pd.Series(sorted_income_temp)],axis=1)
            A_temp=pd.concat([A_temp,pd.Series(sorted_A_temp)],axis=1)

        income_s.append(income_temp.iloc[:,1:])
        A_s.append(A_temp.iloc[:,1:])
        
    # 将排序后的结果降维为300*120的矩阵
    income_s_i=[]
    A_s_i=[]
    income_s_i=pd.Series(income_s_i)
    A_s_i=pd.Series(A_s_i)
    for i in range(20):
        for j in range(6):
            income_s_i=pd.concat([income_s_i,pd.Series(np.array(income_s[i])[:,j].tolist())],axis=1)
            A_s_i=pd.concat([A_s_i,pd.Series(np.array(A_s[i])[:,j].tolist())],axis=1)
    income_s_i=income_s_i.iloc[:,1:]
    A_s_i=A_s_i.iloc[:,1:]
    return A_s_i,income_s_i


# 2.选出单因子优势组合；（比较 5 个组的收益率均值，收益率均值最大的一组为该因子的优势组合）
# 选出单因子优势组合函数choose_superiority，输入：待选择income函数，二维list（300*120）
def choose_superiority(income_s_i):
    for i in range(5):
        m_temp=[]
        m_temp=income_s_i[i*60:60+i*60]
        print('第'+str(i+1)+'组单因子对应平均收益率为'+str(m_temp.mean().mean()))

# 3.配对样本t检验函数，单因子优势组合的收益率为变量income_s_cd
def ttest_superiority(income_s_cd):
    print('配对样本t检验的结果为：'+str(ttest_rel(income_s_cd.mean().values.tolist(),CSI_Index.values.tolist()[0])))


# 计算平均收益率与沪深300均值差值 输入list：单因子优势组合的收益率为变量income_s_cd，返回超额收益率
def mean_difference(income_s_cd):
    excess_rate=income_s_cd.mean().values-CSI_Index.values
    print('D_HS(优势组合均值-沪深300均值:):'+str(income_s_cd.mean().mean()-CSI_Index.mean().mean()))
    return excess_rate


# 4.统计单因子优势组合获得超额收益率的频率；（提示：样本时间跨度为 120 个月，计算获得超额收益率的频率）
# 输入为超额收益率
def freq(excess_rate):
    count=0
    for i in range(120):
        if excess_rate[0][i]>=0:
                count+=1
    freq=count/120
    print('单因子优势组合获得超额收益率的频率:'+str(freq))

# 5.计算组合序数与收益率排名的斯皮尔曼相关系数
# 皮尔逊相关系数要求数据符合正态分布，且实验数据之间的差值不宜过大。
# 斯皮尔曼相关系数常常用于定序数据
# 肯德尔相关系数要求数据是类别数据或者可以分类的数据,针对无序序列、非正态分布的数据。
# 使用corr计算斯皮尔曼相关系数，输入list：组合序数与收益率排名：
def corr_rank(Combination_ordina,income_ranking):
    corr_data=pd.DataFrame([Combination_ordina,income_ranking]).T
    print('计算斯皮尔曼相关系数结果为：'+str(corr_data.corr(method='spearman')))


# %%
# 规模因子-总市值
print('----------------------规模因子-总市值:----------------------')
# 读取市值
total_market_capitalization=[]
for i in range(20):
    total_market_capitalization.append(pd.read_excel("沪深300成分股票规模因子.xlsx",sheet_name=i,header=None).iloc[1:301,3:9])
total_market_capitalization_s_i,income_s_i=rank(total_market_capitalization,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
income_s_cd_1=income_s_i[:60]
# # income_s_cd_2=income_s_i[60:120]
# # income_s_cd_3=income_s_i[120:180]
# # income_s_cd_4=income_s_i[180:240]
# # income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_1)
excess_rate=mean_difference(income_s_cd_1)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[1,3,2,4,2]
corr_rank(Combination_ordina,income_ranking)


# 规模因子-流通市值
print('----------------------规模因子-流通市值:----------------------')
Circulating_market_capitalization=[]
for i in range(20):
    Circulating_market_capitalization.append(pd.read_excel("沪深300成分股票规模因子.xlsx",sheet_name=i,header=None).iloc[1:301,13:19])
# Circulating_market_capitalization
Circulating_market_capitalization_s_i,income_s_i=rank(Circulating_market_capitalization,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
income_s_cd_1=income_s_i[:60]
# # income_s_cd_2=income_s_i[60:120]
# # income_s_cd_3=income_s_i[120:180]
# # income_s_cd_4=income_s_i[180:240]
# # income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_1)
excess_rate=mean_difference(income_s_cd_1)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[1,2,3,4,5]
corr_rank(Combination_ordina,income_ranking)

# %%
# 价值因子-市盈率
print('----------------------价值因子-市盈率:----------------------')
PE=[]
for i in range(20):
    PE.append(pd.read_excel("沪深300成分股票价值因子.xlsx",sheet_name=i,header=None).iloc[1:301,3:9])
# PE

PE_s_i,income_s_i=rank(PE,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
# income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_2)
excess_rate=mean_difference(income_s_cd_2)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[2,1,4,3,5]
corr_rank(Combination_ordina,income_ranking)

# 价值因子-市净率
print('----------------------价值因子-市净率:----------------------')
PB=[]
for i in range(20):
    PB.append(pd.read_excel("沪深300成分股票价值因子.xlsx",sheet_name=i,header=None).iloc[1:301,13:19])
# PB

PB_s_i,income_s_i=rank(PB,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
# income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_1)
excess_rate=mean_difference(income_s_cd_1)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[1,2,3,5,4]
corr_rank(Combination_ordina,income_ranking)

# 价值因子-市现率
print('----------------------价值因子-市现率:----------------------')
PCF=[]
for i in range(20):
    PCF.append(pd.read_excel("沪深300成分股票价值因子.xlsx",sheet_name=i,header=None).iloc[1:301,23:29])
# PCF

PCF_s_i,income_s_i=rank(PCF,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
# income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_2)
excess_rate=mean_difference(income_s_cd_2)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[3,1,2,4,5]
corr_rank(Combination_ordina,income_ranking)

# 价值因子-市销率
print('----------------------价值因子-市销率:----------------------')
PS=[]
for i in range(20):
    PS.append(pd.read_excel("沪深300成分股票价值因子.xlsx",sheet_name=i,header=None).iloc[1:301,33:39])
# PS

PS_s_i,income_s_i=rank(PS,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
# income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_2)
excess_rate=mean_difference(income_s_cd_2)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[2,1,3,4,5]
corr_rank(Combination_ordina,income_ranking)

# %%
# 成长因子-营业收入增长率
print('----------------------成长因子-营业收入增长率:----------------------')
income_growth_rate=[]
for i in range(20):
    income_growth_rate.append(pd.read_excel("沪深300成分股票成长因子.xlsx",sheet_name=i,header=None).iloc[1:301,3:9])
# income_growth_rate

income_growth_rate_s_i,income_s_i=rank(income_growth_rate,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_5)
excess_rate=mean_difference(income_s_cd_5)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[5,4,3,2,1]
corr_rank(Combination_ordina,income_ranking)

# 成长因子-营业利润增长率
print('----------------------成长因子-营业利润增长率:----------------------')
profit_growth_rate=[]
for i in range(20):
    profit_growth_rate.append(pd.read_excel("沪深300成分股票成长因子.xlsx",sheet_name=i,header=None).iloc[1:301,13:19])
# profit_growth_rate

profit_growth_rate_s_i,income_s_i=rank(profit_growth_rate,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_5)
excess_rate=mean_difference(income_s_cd_5)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[5,4,3,2,1]
corr_rank(Combination_ordina,income_ranking)

# 成长因子-净利润增长率
print('----------------------成长因子-净利润增长率:----------------------')
net_profit_growth_rate=[]
for i in range(20):
    net_profit_growth_rate.append(pd.read_excel("沪深300成分股票成长因子.xlsx",sheet_name=i,header=None).iloc[1:301,23:29])
# net_profit_growth_rate

net_profit_growth_rate_s_i,income_s_i=rank(net_profit_growth_rate,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_5)
excess_rate=mean_difference(income_s_cd_5)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[5,4,3,2,1]
corr_rank(Combination_ordina,income_ranking)

# 成长因子-经营活动产生的现金流量增长率
print('----------------------成长因子-经营活动产生的现金流量增长率:----------------------')
cash_flow_growth_rate=[]
for i in range(20):
    cash_flow_growth_rate.append(pd.read_excel("沪深300成分股票成长因子.xlsx",sheet_name=i,header=None).iloc[1:301,33:39])
# cash_flow_growth_rate

cash_flow_growth_rate_s_i,income_s_i=rank(cash_flow_growth_rate,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_5)
excess_rate=mean_difference(income_s_cd_5)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[3,5,4,2,1]
corr_rank(Combination_ordina,income_ranking)

# %%
# 动量因子-前一月涨跌幅
print('----------------------动量因子-前一月涨跌幅:----------------------')
one_month_change=[]
for i in range(20):
    one_month_change.append(pd.read_excel("沪深300成分股票1月涨跌幅.xlsx",sheet_name=i,header=None).iloc[1:301,3:9])
# one_month_change

one_month_change_s_i,income_s_i=rank(one_month_change,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
# income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_2)
excess_rate=mean_difference(income_s_cd_2)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[2,1,3,4,5]
corr_rank(Combination_ordina,income_ranking)

# 动量因子-前三月涨跌幅
print('----------------------动量因子-前三月涨跌幅:----------------------')
three_month_change=[]
for i in range(20):
    three_month_change.append(pd.read_excel("沪深300成分股票3月涨跌幅.xlsx",sheet_name=i,header=None).iloc[2:302,3:9])
# three_month_change

three_month_change_s_i,income_s_i=rank(three_month_change,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
# income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_1)
excess_rate=mean_difference(income_s_cd_1)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[1,3,2,4,5]
corr_rank(Combination_ordina,income_ranking)

# 动量因子-前六月涨跌幅
print('----------------------动量因子-前六月涨跌幅:----------------------')
six_month_change=[]
for i in range(20):
    six_month_change.append(pd.read_excel("沪深300成分股票6月涨跌幅.xlsx",sheet_name=i,header=None).iloc[2:302,3:9])
# six_month_change

six_month_change_s_i,income_s_i=rank(six_month_change,income)
choose_superiority(income_s_i)

# # 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
# income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_1)
excess_rate=mean_difference(income_s_cd_1)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[1,3,4,2,5]
corr_rank(Combination_ordina,income_ranking)


# %%
# 机构预测因子-一致预测净利润
print('----------------------机构预测因子-一致预测净利润:----------------------')
forecast_profit=[]
for i in range(20):
    forecast_profit.append(pd.read_excel("沪深300成分股票分析师预测因子.xlsx",sheet_name=i,header=None).iloc[1:301,3:9])
# forecast_profit

forecast_profit_s_i,income_s_i=rank(forecast_profit,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
income_s_cd_4=income_s_i[180:240]
# income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_4)
excess_rate=mean_difference(income_s_cd_4)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[3,4,2,1,5]
corr_rank(Combination_ordina,income_ranking)

# 机构预测因子-一致预测每股收益EPS 
print('----------------------机构预测因子-一致预测每股收益EPS:----------------------')
forecast_EPS=[]
for i in range(20):
    forecast_EPS.append(pd.read_excel("沪深300成分股票分析师预测因子.xlsx",sheet_name=i,header=None).iloc[1:301,13:19])
# forecast_EPS

forecast_EPS_s_i,income_s_i=rank(forecast_EPS,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_5)
excess_rate=mean_difference(income_s_cd_5)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[5,3,4,2,1]
corr_rank(Combination_ordina,income_ranking)

# %%
# 权益因子-净资产收益率ROE
print('----------------------权益因子-净资产收益率ROE:----------------------')
equity_ROE=[]
for i in range(20):
    equity_ROE.append(pd.read_excel("沪深300成分股票分析师预测因子.xlsx",sheet_name=i,header=None).iloc[1:301,23:29])
# equity_ROE

equity_ROE_s_i,income_s_i=rank(equity_ROE,income)
choose_superiority(income_s_i)

# 切片优势单因子
# 将整合排序后的收益率income_s_i分5段
# income_s_cd_1=income_s_i[:60]
# income_s_cd_2=income_s_i[60:120]
# income_s_cd_3=income_s_i[120:180]
# income_s_cd_4=income_s_i[180:240]
income_s_cd_5=income_s_i[240:]
ttest_superiority(income_s_cd_5)
excess_rate=mean_difference(income_s_cd_5)
freq(excess_rate)
Combination_ordina=[1,2,3,4,5]
income_ranking=[5,3,4,2,1]
corr_rank(Combination_ordina,income_ranking)


