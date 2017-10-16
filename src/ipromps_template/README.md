# Interaction ProMP

This package is the implement of Interaction ProMP described [here](http://www.ausy.tu-darmstadt.de/uploads/Team/PubGJMaeda/phase_estim_IJRR.pdf).

# Requirements

- python >=2.6
- numpy
- sklearn
- scipy >= 0.19.1

# exaple
`load_data.py` : load the data from csv file and resample the time sequence as same duration
`train_offline`: train the Interaction ProMPs from the demonstrations
`test_online`: test the trained models 
