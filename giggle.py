import numpy as np

import matlab.engine
eng = matlab.engine.start_matlab()
print("Started MATLAB")
print(eng.main())


#d = eng.Dummy(3).doSmt()
#print(d)

new_obs = eng.computeNewL(-1, [-2, 1, np.pi/2])