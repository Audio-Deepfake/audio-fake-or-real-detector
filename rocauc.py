#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:45:48 2021

@author: jass
"""

#%%
result = []
#%%
from cnn_model import model as m_cnn
cnn = m_cnn()
result.append(cnn)

#%%
from mlp_based_nn_model import model as m_mlp
mlp = m_mlp()
result.append(mlp)


#%%
from RNN_LSTM_based_model import model as m_rnn
rnn = m_rnn()
result.append(rnn)

#%%
import matplotlib.pyplot as plt
import numpy as np
for i in range(0,len(result)):
    plt.plot(result[i][0], 
             result[i][1], 
             label="{}, AUC={:.3f}".format(result[i][3], result[i][2]))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.savefig("dataset_2",dpi=300, bbox_inches='tight')
plt.show()
