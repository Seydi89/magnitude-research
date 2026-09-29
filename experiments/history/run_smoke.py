#!/usr/bin/env python3
import numpy as np
from magnitude_research.history import weighted_history_query

current=np.array([0.,1.,0.]); prior=[np.array([1.,0.,0.]),np.array([.8,.2,0.])]
print({"current":current.tolist(),"history_query":weighted_history_query(current,prior,.7).tolist()})

