#!/usr/bin/env python3
import numpy as np
from magnitude_research.history import transition_weights

scales=np.geomspace(1/64,16,17)
for name,changes in {"stable":[0,0],"narrowing":[.2,.8],"switch":[.9,.9]}.items():
    w=transition_weights(np.asarray(changes),scales)
    print(name,"expected_scale",float(w@scales))

