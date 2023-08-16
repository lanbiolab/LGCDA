from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp


from similarty_calculated import similarity





def mydataset_create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None,
                              datasplit_from_file=False, verbose=True, rating_map=None,
                              post_rating_map=None, ratio=1.0):

    num_users = 514
    num_items = 62

    train_exist_users = []
    train_exist_items = []
    trainfile_path = '.\\mydataset\\dataset2\\fold_file\\circRNA-disease-fold5\\train.txt' #某一数据集的某一折训练数据路径
    testfile_path = '.\\mydataset\\dataset2\\fold_file\\circRNA-disease-fold5\\test.txt'


    circRNA_similarity, disease_similarity = similarity(num_users, num_items, trainfile_path)

    with open(trainfile_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                train_exist_users.append(uid)
                train_exist_items.append(items)


    train_u_nodes = []
    train_v_nodes = []
    train_ratings = []
    train_index = []

    trainall_u_nodes = []
    trainall_v_nodes = []
    trainall_ratings = []
    trainall_index = []



    for i in range(len(train_exist_users)):
        temp_user = train_exist_users[i]
        for j in range(len(train_exist_items[i])):
            temp_item = train_exist_items[i][j]
            train_u_nodes.append(temp_user)
            train_v_nodes.append(temp_item)
            train_ratings.append(1)
            train_index.append((temp_user,temp_item))

            trainall_u_nodes.append(temp_user)
            trainall_v_nodes.append(temp_item)
            trainall_ratings.append(1)
            trainall_index.append((temp_user, temp_item))





    np.random.seed(116)
    np.random.shuffle(train_u_nodes)
    np.random.seed(116)
    np.random.shuffle(train_v_nodes)
    np.random.seed(116)
    np.random.shuffle(train_ratings)
    np.random.seed(116)
    np.random.shuffle(train_index)


    test_exist_users = []
    test_exist_items = []

    with open(testfile_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                test_exist_users.append(uid)
                test_exist_items.append(items)


    test_u_nodes = []
    test_v_nodes = []
    test_ratings = []
    test_index = []

    for i in range(len(test_exist_users)):
        temp_user = test_exist_users[i]
        for j in range(len(test_exist_items[i])):
            temp_item = test_exist_items[i][j]
            test_u_nodes.append(temp_user)
            test_v_nodes.append(temp_item)
            test_ratings.append(1)
            test_index.append((temp_user, temp_item))


    temp_train_index = train_index.copy()
    for (u,i) in temp_train_index:
        for num in range(4):
            j = np.random.randint(num_items)
            while (u,j) in train_index:
                j = np.random.randint(num_items)

            trainall_u_nodes.append(u)
            trainall_v_nodes.append(j)
            trainall_ratings.append(0)
            trainall_index.append((u, j))


    nozero_test_index = test_index.copy()
    for i in range(num_users):
        for j in range(num_items):
            if (i, j) not in train_index and (i, j) not in nozero_test_index:
                test_u_nodes.append(i)
                test_v_nodes.append(j)
                test_ratings.append(0)
                test_index.append((i, j))






    np.random.seed(116)
    np.random.shuffle(test_u_nodes)
    np.random.seed(116)
    np.random.shuffle(test_v_nodes)
    np.random.seed(116)
    np.random.shuffle(test_ratings)
    np.random.seed(116)
    np.random.shuffle(test_index)

    np.random.seed(116)
    np.random.shuffle(trainall_u_nodes)
    np.random.seed(116)
    np.random.shuffle(trainall_v_nodes)
    np.random.seed(116)
    np.random.shuffle(trainall_ratings)
    np.random.seed(116)
    np.random.shuffle(trainall_index)


    u_features, v_features = None, None

    train_ratings = np.array(train_ratings)
    test_ratings = np.array(test_ratings)

    trainall_ratings = np.array(trainall_ratings)



    num_val = int(np.ceil(train_ratings.shape[0] * 0.05))
    train_pairs_nonzero = np.vstack([train_u_nodes, train_v_nodes]).transpose()
    test_pairs_nonzero = np.vstack([test_u_nodes, test_v_nodes]).transpose()
    trainall_pairs_nonzero = np.vstack([trainall_u_nodes, trainall_v_nodes]).transpose()
    num_train = train_ratings.shape[0] - num_val

    train_pairs_idx = train_pairs_nonzero.copy()
    val_pairs_idx = train_pairs_nonzero[num_train:]
    test_pairs_idx = test_pairs_nonzero.copy()
    trainall_pairs_idx = trainall_pairs_nonzero.copy()

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()
    u_trainall_idx, v_trainall_idx = trainall_pairs_idx.transpose()




    if rating_map is not None:
        for i, x in enumerate(train_ratings):
            train_ratings[i] = rating_map[x]


    if rating_map is not None:
        for i, x in enumerate(test_ratings):
            test_ratings[i] = rating_map[x]


    train_rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(test_ratings)).tolist())}
    test_rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(test_ratings)).tolist())}
    trainall_rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(trainall_ratings)).tolist())}



    train_all_labels = np.array([train_rating_dict[r] for r in train_ratings], dtype=np.int32)
    test_all_labels = np.array([test_rating_dict[r] for r in test_ratings], dtype=np.int32)
    trainall_all_labels = np.array([trainall_rating_dict[r] for r in trainall_ratings], dtype=np.int32)

    train_labels = train_all_labels.copy()

    val_labels = train_all_labels[num_train:]
    test_labels = test_all_labels.copy()
    trainall_labels = trainall_all_labels.copy()

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])


    class_values = np.sort(np.unique(test_ratings))


    data = train_labels
    data = data.astype(np.float32)


    rating_mx_train = sp.csr_matrix((data, [u_train_idx, v_train_idx]),
                                    shape=[num_users, num_items], dtype=np.float32)


    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, trainall_labels, u_trainall_idx, v_trainall_idx, class_values, circRNA_similarity, disease_similarity


