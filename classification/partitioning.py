import numpy as np

def partition_by_class(labels, label_test, seed=None):
    unique_labels = np.unique(labels)
    unique_test_labels = np.unique(label_test)
    train_partition = []
    test_partition = []
    np.random.seed(seed)
    for lab in unique_labels:
        if lab in unique_test_labels:
            rows = np.where(labels == lab)[0]
            rows = np.random.permutation(rows)
            test_partition += list(rows)
        else:
            rows = np.where(labels == lab)[0]
            rows = np.random.permutation(rows)
            train_partition += list(rows)
    return np.random.permutation(train_partition), \
           np.random.permutation(test_partition)


def _partition_stratified(labels, train_size, seed=None):
    unique_labels = np.unique(labels)
    train_partition = []
    test_partition = []
    np.random.seed(seed)
    for lab in unique_labels:
        rows = np.where(labels == lab)[0]
        rows = np.random.permutation(rows)
        lab_size = np.sum(labels == lab)
        row_end = np.int(lab_size * train_size)
        train_partition += list(rows[:row_end])
        row_beg = row_end
        test_partition += list(rows[row_beg:])
    return np.random.permutation(train_partition), \
           np.random.permutation(test_partition)

def _partition_stratified_validation(labels, train_size, valid_size, seed=None):
    unique_labels = np.unique(labels)
    train_partition = []
    validation_partition = []
    test_partition = []
    for lab in unique_labels:
        rows = np.where(labels == lab)[0]
        lab_size = np.sum(labels == lab)
        row_end = np.int(lab_size * train_size)
        train_partition += list(rows[:row_end])
        row_beg = row_end
        row_end = np.int(lab_size*(train_size + valid_size))
        validation_partition += list(rows[row_beg:row_end])
        row_beg = row_end
        test_partition += list(rows[row_beg:])
    np.random.seed(seed)
    return np.random.permutation(train_partition), \
           np.random.permutation(validation_partition), \
           np.random.permutation(test_partition)

def divide_in_kparts(v, K):
    lenv = len(v)
    stepk = np.int(lenv / K)
    parts = []
    for k in range(K-1):
        parts += [v[k*stepk : (k+1)*stepk]]
    parts += [v[(K-1)*stepk:]]
    return parts

def cpart(partition):
    counts = np.zeros(len(partition))
    for n in range(len(partition)):
        counts[n] = len(partition[n])
        if counts[n] == 0:
            counts[n] = 1e-7
    return counts / np.sum(counts)

def portions(part, labels):
    un_labels = np.unique(labels)
    portion = np.zeros(len(un_labels))
    for n in range(len(un_labels)):
        portion[n] = np.sum(labels[part] == un_labels[n])
        if portion[n] == 0:
            portion[n] = 1e-7
    portion = portion / (np.sum(portion))
    return portion

def entropy(portion):
    ent = 0
    for n in range(len(portion)):
        ent = ent - portion[n] * np.log(portion[n])
    return ent

def cross_entropy(portion, target):
    ent = 0
    for n in range(len(portion)):
        ent = ent - target[n] * np.log(portion[n])
    return ent

def epart(partition, labels):
    entropies = np.zeros(len(partition))
    for n in range(len(partition)):
        ports = portions(partition[n], labels)
        entropies[n] = entropy(ports)
    return entropies

def pargmax(partition, crows, labels, coeff=1.0):
    partition_ = partition.copy()
    vals = np.zeros(len(partition))


    for n in range(len(partition)):
        partition_[n] = np.append(partition_[n], crows)
        entropies_ = epart(partition_, labels)
        counts_ = cpart(partition_)
        vals[n] = entropies_.mean() + entropy(counts_)*coeff
        partition_ = partition.copy()
        #print(entropies_.sum(), entropy(counts_))
    #print(vals)
    return np.argmax(vals)

def pargmax_cross(partition, crows, labels, target_dist, coeff=1.0):
    partition_ = partition.copy()
    vals = np.zeros(len(partition))
    for n in range(len(partition)):
        partition_[n] = np.append(partition_[n], crows)
        entropies_ = epart(partition_, labels)
        counts_ = cpart(partition_)
        vals[n] = entropies_.mean() - cross_entropy(counts_, target_dist)*coeff
        partition_ = partition.copy()
        #print(entropies_.sum(), entropy(counts_))
    #print(vals)
    return np.argmax(vals)


### THE NEXT TWO FUNCTIONS NEED TO BE TESTED.
def partition_stratified_validation(labels, train_size, valid_size, seed=None, clabels=None):
    if clabels is None:
        return _partition_stratified_validation(labels, train_size, valid_size, seed=None)

    print('This might take a while - get a coffeeee.')
    unique_clabels = np.unique(clabels)
    np.random.seed(seed)
    partition = list(np.zeros((3,1)))
    target_dist = np.array([train_size, valid_size, 1.0-train_size-valid_size])
    num_unique_labels = len(np.unique(labels))
    coeff = entropy(np.ones(num_unique_labels)*1./num_unique_labels)/entropy(target_dist)
    for k in range(3):
        partition[k] = np.array([],dtype=np.uint16)

    for clab, m in zip(unique_clabels, range(len(unique_clabels))):
        crows = np.where(clabels == clab)[0]
        if m == 0:
            partition[0] = np.append(partition[0], crows)
        else:
            mpart = pargmax_cross(partition, crows, labels, target_dist, coeff=coeff*5)
            partition[mpart] = np.append(partition[mpart], crows)
        if m % 50 == 0:
            print('{} / {} done'.format(m, len(unique_clabels)))
            print('cparts: {}'.format(cpart(partition)))
    for k in range(3):
        partition[k] = np.random.permutation(partition[k])
    return partition

def partition_stratified(labels, train_size, seed=None, clabels=None, label_test=None):
    if clabels is None:
        return _partition_stratified(labels, train_size, seed=None)

    if label_test is not None:
        return partition_by_class(K, labels, seed=None, label_test=label_test)

    print('This might take a while - get a coffeeee.')
    unique_clabels = np.unique(clabels)
    np.random.seed(seed)
    partition = list(np.zeros((2,1)))
    target_dist = np.array([train_size, 1.0-train_size])
    num_unique_labels = len(np.unique(labels))
    coeff = entropy(np.ones(num_unique_labels)*1./num_unique_labels)/entropy(target_dist)
    for k in range(2):
        partition[k] = np.array([],dtype=np.uint16)

    for clab, m in zip(unique_clabels, range(len(unique_clabels))):
        crows = np.where(clabels == clab)[0]
        if m == 0:
            partition[0] = np.append(partition[0], crows)
        else:
            mpart = pargmax_cross(partition, crows, labels, target_dist, coeff=coeff*5)
            partition[mpart] = np.append(partition[mpart], crows)
        if m % 50 == 0:
            print('{} / {} done'.format(m, len(unique_clabels)))
            print('cparts: {}'.format(cpart(partition)))
    for k in range(2):
        partition[k] = np.random.permutation(partition[k])
    return partition

def partition_stratified_kfold(K, labels, seed=None, clabels=None):
    if clabels is None:
        return _partition_stratified_kfold(K, labels, seed=None)
    
    print('This might take a while - get a coffeeee.')
    unique_clabels = np.unique(clabels)
    np.random.seed(seed)
    partition = list(np.zeros((K,1)))
    num_unique_labels = len(np.unique(labels))
    coeff = entropy(np.ones(num_unique_labels)*1./num_unique_labels)/entropy(np.ones(K)*1./K)
    for k in range(K):
        partition[k] = np.array([],dtype=np.uint16)

    for clab, m in zip(unique_clabels, range(len(unique_clabels))):
        crows = np.where(clabels == clab)[0]
        if m == 0:
            partition[0] = np.append(partition[0], crows)
        else:
            mpart = pargmax(partition, crows, labels, coeff=coeff*5)
            partition[mpart] = np.append(partition[mpart], crows)
        if m % 50 == 0:
            print('{} / {} done'.format(m, len(unique_clabels)))
            print('cparts: {}'.format(cpart(partition)))
    for k in range(K):
        partition[k] = np.random.permutation(partition[k])
    return partition

def _partition_stratified_kfold(K, labels, seed=None):
    unique_labels = np.unique(labels)
    np.random.seed(seed)
    for lab, n in zip(unique_labels, range(len(unique_labels))):
        rows = np.where(labels == lab)[0]
        rows = np.random.permutation(rows)
        kfold_parts = divide_in_kparts(rows, K)
        if n == 0:
            partitions = kfold_parts
        else:
            for k in range(K):
                partitions[k] = np.append(partitions[k], kfold_parts[k])
    for k in range(K):
        partitions[k] = np.random.permutation(partitions[k])
    return partitions

def get_partition_stratified_kfold(kp, partitions):
    test_partition = partitions[kp]
    train_partition = []
    for k in range(len(partitions)):
        if k != kp:
            train_partition += list(partitions[k])
    return train_partition, test_partition

def decimate_partition_stratified(current_partition, labels, psize=1.0):
    train_part, _ = partition_stratified(labels[current_partition], train_size=psize)
    return list(np.asarray(current_partition)[train_part])

def partition(labels, train_size, seed=None):
    train_partition = []
    validation_partition = []
    test_partition = []
    np.random.seed(seed)
    rows = np.random.permutation(labels.size)
    lab_size = labels.size
    row_end = np.int(lab_size * train_size)
    train_partition += list(rows[:row_end])
    row_beg = row_end
    test_partition += list(rows[row_beg:])
    return train_partition, test_partition

def combine_partitions(init_part, groups):
    ngroups = len(groups)
    fin_part = []
    for n in range(ngroups):
        part_ = np.array([], dtype=np.uint16)
        for g in groups[n]:
            part_ = np.append(part_, init_part[g])
        fin_part = fin_part + [part_]
    return fin_part
