import numpy as np

M = np.array([[1, 2, 3], 
			[3, 4, 5], 
			[5, 4, 3],
			[np.nan, 2, 4],
			[1, 3, 5]])

def inverseProcess(matrix, mean_user, mean_item):
    nM = matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                continue
            else:
                s = mean_user[i] + mean_item[j]
                nM[i][j] += s
    return nM

def preprocessing(matrix):
    mean_user = np.mean(matrix, axis=1)
    mean_item = np.mean(matrix, axis=0)
    nM = matrix - mean_user[:, np.newaxis]
    nM = nM - mean_item[np.newaxis, :]
    return nM, mean_user, mean_item


mask = np.isnan(M)
temp_mask = np.ma.masked_array(M, mask) # (movies x users)
nM, mean_user, mean_item = preprocessing(temp_mask)
print(nM)

nM = nM.filled(np.zeros(temp_mask.shape[1]) )

print(inverseProcess(nM, mean_user, mean_item))
