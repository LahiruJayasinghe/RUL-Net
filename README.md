# RUL-Net
Deep learning approach for estimation of Remaining Useful Life (RUL) of an engine
This repo is dedicated to new architectures for estimating RUL using CMAPSS dataset and PHM08 prognostic challenge dataset
The datasets are included in this repo or can be donwloaded from: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan 

(This is an undergoing project, more new things are yet to come)

## Dataset discription
The main and largest dataset is the CMAPSS dataset. It contain four subdatasets called FD001, FD002, FD003, and FD004

| Dataset | Train trajectories | Test trajectories | Conditions | Fault Modes |
| ------- | ------------------ | ----------------- | ---------- | ----------- |
| FD001 | 100 | 100 | ONE (Sea Level) | ONE (HPC Degradation) |
| FD002 | 260 | 259 | SIX | ONE (HPC Degradation) |
| FD003 | 100 | 100 | ONE (Sea Level) | TWO (HPC Degradation, Fan Degradation) |
| FD004 | 248 | 249 | SIX (Sea Level) | TWO (HPC Degradation, Fan Degradation) |

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine – i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

### Dataset structure 

The data provieded as text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  26

![Screenshot](screenshots/CMAPSSData_snapshot.PNG)

After standard normalization of data in time domain 

| Dataset | Visualization |
| ------- | --------------|
| FD001 and FD003| ![Screenshot](screenshots/training_data_plot0103.PNG) |
| FD002 and FD004| ![Screenshot](screenshots/training_data_plot0204.PNG) |

The distribution of FD001 and FD003 has similarities as well as the FD002 and FD004 datasets.
## Data labeling


## Dependencies 
