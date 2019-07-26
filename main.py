#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:32:32 2019

@author: jyc
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from get_subway_data import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def main():
    # Get Subway Data
    # df_i, df_o = get_subway_data(2016, 2018)
    df_i = pd.read_csv('rawdata/subway_in.csv', encoding='utf8')
    df_o = pd.read_csv('rawdata/subway_out.csv', encoding='utf8')
    name2en = pd.read_csv('rawdata/subway_en.csv', encoding='utf8')
    for df in [df_i, df_o]:

        stop_list = df.columns[3:]
        df[stop_list] = StandardScaler().fit_transform(df[stop_list].values)

        # Distrotion Plot
        figN = 1
        fig, ax = plt.subplots()
        for s in np.random.random_integers(0, len(df)-8, 4):
            distortions = []
            for k in range(1, 9):
                km = KMeans(n_clusters=k,
                            init='random',
                            n_init=10,
                            max_iter=300,
                            random_state=0)
                km.fit(df.loc[s:s+7, stop_list].transpose())
                distortions.append(km.inertia_)
            plt.subplot(2, 2, figN)
            plt.plot(range(1, 9), distortions, marker='o')
            plt.title('Clustering w data start from ' +
                      df.loc[s][['Year', 'Month', 'Day']].apply(int).apply(str).str.cat(sep='/'))
            plt.xlabel('Number of clusters')
            plt.ylabel('Distortion')
            figN += 1
        plt.tight_layout(pad=0)
        plt.show()
        fig.set_size_inches(11, 6)
        fig.savefig(('Entry' if df is df_i else 'Exit')+'Distortion.png', dpi=500)
        
        # Choose 3 as number of clustering group
        km = KMeans(n_clusters=3,
                    init='random',
                    n_init=30,
                    max_iter=300,
                    tol=1e-04,
                    random_state=0)
        clustering_table = pd.DataFrame(columns=stop_list)
        for d in range(len(df)//7):
            clustering_table.loc[d] = km.fit_predict(df.loc[d*7:d*7+7, stop_list].transpose())

        stop_sim_table = [[np.mean(clustering_table[y] == clustering_table[x])
                           * 100 for x in stop_list] for y in stop_list]
        stop_sim_table = pd.DataFrame(stop_sim_table, columns=stop_list).set_index(stop_list)
        stop_sim_mean = stop_sim_table.mean().sort_values(ascending=False).index
        stop_sim_mean_en = np.array(
            [np.array(name2en[name2en['Name'] == x]['Name_en'])[0] for x in stop_sim_mean])
        stop_sim_mean_n = np.array(
            [np.array(name2en[name2en['Name'] == x]['Number'])[0] for x in stop_sim_mean])
        stop_sim_table = stop_sim_table.reindex(
            stop_sim_mean, axis=0).reindex(stop_sim_mean, axis=1)

        fig, ax = plt.subplots()
        image = ax.imshow(stop_sim_table, cmap='Blues')
        ax.set_xticks(np.arange(18)*6+3)
        ax.set_yticks(np.arange(18)*6+3)
        ax.set_xticklabels(stop_sim_mean_n[np.arange(0, len(stop_list), 6)+3])
        ax.set_yticklabels(stop_sim_mean_n[np.arange(0, len(stop_list), 6)+3])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title('Similarity within Each Stop with ' +
                     ('Entry' if df is df_i else 'Exit')+' data')
        ax.set_xlabel('Stop Numbering')
        ax.set_ylabel('Stop Numbering')
        fig.colorbar(image, label='Similarity (%)')
        fig.tight_layout()
        plt.show()
        fig.set_size_inches(10, 9)
        fig.savefig(('Entry' if df is df_i else 'Exit')+'SimilarityHeatMap.png', dpi=500)

        # Seperate into 3 groups
        if df is df_i:
            group1 = stop_sim_mean_en[0:79]
            group2 = stop_sim_mean_en[79:91]
            group3 = stop_sim_mean_en[91:109]
        else:
            group1 = stop_sim_mean_en[0:79]
            group2 = stop_sim_mean_en[79:93]
            group3 = stop_sim_mean_en[93:109]
        print('<Group 1>\n', group1)
        print('<Group 2>\n', group2)
        print('<Group 3>\n', group3)


if __name__ == '__main__':
    main()
