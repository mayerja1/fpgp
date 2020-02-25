import csv
import numpy as np

if __name__ == '__main__':

    with open('yacht_hydrodynamics.csv', 'r', newline='') as f:
        r = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC, delimiter=' ')
        data = []
        #next(r)
        for line in r:
            data.append(line)
        data = np.array(data[:-1])

        # 70% of the data is training set and 30% test set
        trn_sz = int(len(data) * 0.7)
        tst_sz = len(data) - trn_sz
        indices = np.arange(len(data))
        trn_indices = np.random.choice(indices, trn_sz, replace=False)
        tst_indices = np.array([i for i in indices if i not in trn_indices])
        assert(len(trn_indices) + len(tst_indices) == len(data))
        #print(len(trn_indices), len(tst_indices), len(data))

    np.savez('yachts', trn=data[trn_indices], tst=data[tst_indices])
