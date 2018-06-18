import numpy as np
import jmlipman.ml

data = jmlipman.ml.getDataset("mnist")
data = jmlipman.ml.splitDataset(data,[0.8,0.2,0])

for target in range(10):
	idx = np.where(data["train"][1]==target)[0]
	print(target,np.sum(np.std(data["train"][0][idx],axis=0)))
