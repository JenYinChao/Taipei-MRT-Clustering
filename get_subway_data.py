#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:19:46 2019

@author: jyc
"""


def get_subway_data(start_year, end_yaer, start_month=1, end_month=12):
    import pandas as pd
    import numpy as np

    df_i = pd.DataFrame()
    df_o = pd.DataFrame()
    for year in np.arange(start_year, end_yaer+1).astype(str):
        for month in [str(x) if x >= 10 else '0'+str(x) for x in range(start_month, end_month+1)]:
            df_it = pd.read_excel('rawdata/subway/'+year+month+'_cht.xlsx', '進站資料')
            df_ot = pd.read_excel('rawdata/subway/'+year+month+'_cht.xlsx', '出站資料')
            df_i = pd.concat([df_i, df_it], ignore_index=True)
            df_o = pd.concat([df_o, df_ot], ignore_index=True)
    df_col = ['Year', 'Month', 'Day']
    df_col.extend(df_i.columns[1:])
    df_i['Date'] = pd.to_datetime(df_i[df_i.columns[0]])
    df_i['Year'], df_i['Month'], df_i['Day'] = df_i['Date'].dt.year, df_i['Date'].dt.month, df_i['Date'].dt.day
    df_i = df_i[df_col]
    df_o['Date'] = pd.to_datetime(df_o[df_o.columns[0]])
    df_o['Year'], df_o['Month'], df_o['Day'] = df_o['Date'].dt.year, df_o['Date'].dt.month, df_o['Date'].dt.day
    df_o = df_o[df_col]
    df_i.to_csv('rawdata/subway_in.csv', encoding='utf8', index=False, header=df_col)
    df_o.to_csv('rawdata/subway_out.csv', encoding='utf8', index=False, header=df_col)
    return (df_i, df_o)
