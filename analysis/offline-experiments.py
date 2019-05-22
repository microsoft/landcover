
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys, os, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


get_ipython().system('ls /mnt/blobfuse/train-output/offline-active-learning/test1/random')


# In[ ]:


mnt(/blobfuse/analysis/active-learning/test${i}/${strategy}/fine_tune_test_results.csv)

where ${i} in 1, 2, 3, 4
           ${strategy} is "random", "entropy", or "margin"


# In[59]:


test_areas = ["test1","test2","test3","test4"]
strategies = ["random", "entropy", "margin"]

dfs = []

for test_area in test_areas:
    for strategy in strategies:
        fn = "/mnt/blobfuse/train-output/offline-active-learning/%s/%s/fine_tune_test_results.csv" % (
            test_area, strategy
        )
        if os.path.exists(fn):
            try:
                df = pd.read_csv(fn)
                df["strategy"] = strategy
                df[" area"] = test_area
                dfs.append(df)
            except Exception as e:
                print(e, fn)
        else:
            print("%s does not exist" % (fn))
df = pd.concat(dfs)


# In[71]:


methods = [
    'last_k_layers_lr_0.010000_last_k_1',
    'last_k_layers_lr_0.005000_last_k_2',
    'last_k_layers_lr_0.001000_last_k_3',
    'group_params_lr_0.002500'
]
num_points = 400
num_seeds = 5
test_area = "test1"

mean_results = np.zeros((len(test_areas), len(strategies), len(methods), 2), dtype=float)
std_results = np.zeros((len(test_areas), len(strategies), len(methods), 2), dtype=float)
results[:] = np.nan

for i, test_area in enumerate(test_areas):
    for j, strategy in enumerate(strategies):
        for k, method_id in enumerate(methods):

            subset = df[
                (df[" area"] == test_area) &
                (df["strategy"] == strategy) &
                (df["method"] == method_id) &
                (df[" num_points"] == num_points)
            ]
            
            
            mean_results[i,j,k,0] = subset[' pixel_accuracy'].mean()
            mean_results[i,j,k,1] = subset[' mean_IoU'].mean()
            
            std_results[i,j,k,0] = subset[' pixel_accuracy'].std()
            std_results[i,j,k,1] = subset[' mean_IoU'].std()


# In[78]:


for i, test_area in enumerate(test_areas):
    print(test_area)
    print(pd.DataFrame(mean_results[i,:,:,0], index=strategies, columns=methods).to_csv())

