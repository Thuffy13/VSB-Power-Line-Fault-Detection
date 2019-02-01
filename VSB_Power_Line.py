import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

train = pq.read_pandas('C:\\Users\\thuff\\Downloads\\train.parquet').to_pandas()
train.head(40)

plt.figure()
plt.plot(train.iloc[:,-1])

def phase_indices(signal_num):
    phase1 = 3*signal_num
    phase2 = 3*signal_num + 1
    phase3 = 3*signal_num + 2
    return phase1,phase2,phase3

p1, p2, p3 = phase_indices(12)



def low_pass(sig, threshhold = 1e-4):    #threshold can be changed
    fourier = fftpack.rfft(sig)   #returns an n-dimensional array of the real parts of the transformed sig so will be 2 or 3-D
    n = fourier.size
    spacing = 0.002
    freqs = fftpack.rfftfreq(n, d=spacing)
    fourier[freqs > threshold] = 0 #sets high frequency indicies to zero for smoothing
    return fftpack.ifft(fourier)   #returns only the low frequencies


def high_pass(sig, threshold = 1e-7):
    fourier = fftpack.rfft(sig)
    n = fourier.size
    spacing = 0.002
    freqs = fftpack.rfftfreq(n, spacing)
    fourier[freqs < threshold] = 0  #sets low frequencies to zero
    return fftpack.ifft(fourier)


filtered_sig = []
for i in range(8):
    filtered = high_pass(train.iloc[:,i])
    filtered_sig.append(filtered)

plt.figure()
plt.plot(filtered_sig[p1])
#plt.plot(filtered_sig[p2])
#plt.plot(filtered_sig[p3])

#adverserial validation/SMOTE other under-sampling techniques
X_trn, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2,
                                                    random_state = 41)

#SMOTE is an over-sampling tool used for imbalanced data such as this
sm = SMOTE(random_state = 2)

#np.ravel() returns a 1-D arrary for the data which is multi-dimensional I'm assuming
X_train_res, y_train_res = sm.fit_sample(X_trn, y_train.np.ravel())

clf = RandomForestClassifier()
clf.fit(X_train_res, y_train_res)
print(clf.roc_auc_score(X_test, y_test))

