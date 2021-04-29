from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def read_validation_file(path_to_file:str):
    with open(path_to_file, 'r') as file:
        line=file.readline()
        metric_values=defaultdict(list)
        while line !='':
            if line[:6]=='Epoch ':
                line=file.readline()
                while line!='\n':
                    if 'metric' in line:
                        current_metric=line
                    if line[:5]=='value':
                        split_line=line.split('_')
                        value=float(split_line[-1].split(':')[-1])
                        class_num=int(float(split_line[-1].split(':')[0]))
                        metric_values[current_metric].append([class_num, value])
                    line=file.readline()
            else:
                line=file.readline()
        return metric_values

def unpack_metric_values_from_dict(metric_values:Dict[str, List[float]]):
    dataframes={}
    for key, items in metric_values.items():
        values=[]
        for item in items:
            class_num, value=item
            if len(values)<class_num+1:
                values.append([])
            values[class_num].append(value)
        dataframe=pd.DataFrame(data=np.array(values).T, columns=['class_%i'%i for i in range(len(values))])
        dataframes[key]=dataframe
    return dataframes

def rename_metric_names(metric_name:str)->str:
    if 'recall' in metric_name:
        return 'recall'
    if 'accuracy' in metric_name:
        return 'accuracy'
    if 'f1' in metric_name:
        return 'f1'


def plot_lineplots_from_dict_dataframes(dict_dataframes:Dict[str, pd.DataFrame], limits=(0.2, 0.8)):
    plt.ylim(limits[0], limits[1])

    colors=['b', 'g', 'r', 'c', '--b', '--g', '--r', '--c', '.b', '.g', '.r', '.c']
    color_idx=0
    for metric_name,dataframe in dict_dataframes.items():
        for class_type_idx in range(dataframe.shape[1]):
            plt.plot(dataframe.index, dataframe.iloc[:,class_type_idx], colors[color_idx],label='%s,class_type_%i'%(rename_metric_names(metric_name), class_type_idx))
            color_idx+=1
    plt.legend()
    plt.show()

def visualize_validation_file(path_to_validation_file:str)->None:
    metric_values = read_validation_file(path_to_validation_file)
    metric_values = unpack_metric_values_from_dict(metric_values)
    plot_lineplots_from_dict_dataframes(metric_values)



if __name__ == '__main__':
    visualize_validation_file(r'C:\Users\Dresvyanskiy\Desktop\Projects\EngagementRecognition\src\val_logs.txt')

