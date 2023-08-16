import numpy as np

def CircRNA_GIP (association_network, circRNA_network):
    pard = 0
    for i in range(0,association_network.shape[0]):
        pard = pard + np.sum(np.multiply(association_network[i, :], association_network[i, :]))

    pard1 = association_network.shape[0] / pard

    for m in range(0, association_network.shape[0]):
        for n in range(0, association_network.shape[0]):
            minus = association_network[m, :] - association_network[n, :]
            circRNA_network[m, n] = np.exp(-pard1 * np.sum(np.multiply(minus, minus)))

    return circRNA_network


def Disease_GIP (association_network, disease_network):
    pard = 0
    for j in range(0, association_network.shape[1]):
        pard = pard + np.sum(np.multiply(association_network[:, j], association_network[:, j]))

    pard1 = association_network.shape[1] / pard

    for a in range(0, association_network.shape[1]):
        for b in range(0, association_network.shape[1]):
            minus = association_network[:, a] - association_network[:, b]
            disease_network[a, b] = np.exp(-pard1 * np.sum(np.multiply(minus, minus)))

    return disease_network



def circRNA_cosine (association_network, circRNA_network):
    for i in range(0, association_network.shape[0]):
        for j in range(0, association_network.shape[0]):
            numerator = np.sum(np.multiply(association_network[i, :], association_network[j, :]))
            denominator1 = np.sqrt(np.sum(np.multiply(association_network[i, :], association_network[i, :])))
            denominator2 = np.sqrt(np.sum(np.multiply(association_network[j, :], association_network[j, :])))
            denominator = denominator1 * denominator2
            if denominator == 0:
                circRNA_network[i, j] = 0
            else:
                circRNA_network[i, j] = numerator / denominator

    return circRNA_network



def disease_cosine (association_network, disease_network):
    for i in range(0, association_network.shape[1]):
        for j in range(0, association_network.shape[1]):
            numerator = np.sum(np.multiply(association_network[:, i], association_network[:, j]))
            denominator1 = np.sqrt(np.sum(np.multiply(association_network[:, i], association_network[:, i])))
            denominator2 = np.sqrt(np.sum(np.multiply(association_network[:, j], association_network[:, j])))
            denominator = denominator1 * denominator2
            if denominator == 0:
                disease_network[i, j] = 0
            else:
                disease_network[i, j] = numerator / denominator

    return disease_network


def circRNA_similarity_integrated (circRNA_GIP_kernel, circRNA_cos, circRNA_final_similarity):
    for i in range(0, circRNA_cos.shape[0]):
        for j in range(0, circRNA_cos.shape[1]):
            if circRNA_cos[i, j] == 0:
                circRNA_final_similarity[i, j] = circRNA_GIP_kernel[i, j]
            else:
                circRNA_final_similarity[i, j] = circRNA_cos[i, j]

    return circRNA_final_similarity

def disease_similarity_integrated (disease_GIP_kernel, disease_cos, disease_final_similarity):
    for i in range(0,disease_cos.shape[0]):
        for j in range(0, disease_cos.shape[1]):
            if disease_cos[i, j] == 0:
                disease_final_similarity[i, j] = disease_GIP_kernel[i, j]
            else:
                disease_final_similarity[i, j] = disease_cos[i, j]

    return disease_final_similarity

def similarity(num_users, num_items, trainfile_path):
    circrna_disease_matrix = np.zeros((num_users, num_items))
    train_exist_users = []
    train_exist_items = []

    with open(trainfile_path) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                train_exist_users.append(uid)
                train_exist_items.append(items)


    for i in range(len(train_exist_users)):
        temp_user = train_exist_users[i]
        for j in range(len(train_exist_items[i])):
            temp_item = train_exist_items[i][j]
            circrna_disease_matrix[temp_user][temp_item] = 1

    new_circrna_disease_matrix = circrna_disease_matrix.copy()


    circRNA_net = np.mat(np.zeros((new_circrna_disease_matrix.shape[0], new_circrna_disease_matrix.shape[0])))
    disease_net = np.mat(np.zeros((new_circrna_disease_matrix.shape[1], new_circrna_disease_matrix.shape[1])))
    circRNA_cos = circRNA_cosine(new_circrna_disease_matrix, circRNA_net)
    disease_cos = disease_cosine(new_circrna_disease_matrix, disease_net)

    circRNA_GIP = CircRNA_GIP(new_circrna_disease_matrix, circRNA_net)
    disease_GIP = Disease_GIP(new_circrna_disease_matrix, disease_net)


    circRNA_sim = np.mat(np.zeros((new_circrna_disease_matrix.shape[0], new_circrna_disease_matrix.shape[0])))
    disease_sim = np.mat(np.zeros((new_circrna_disease_matrix.shape[1], new_circrna_disease_matrix.shape[1])))
    circRNA_similarity = circRNA_similarity_integrated(circRNA_GIP, circRNA_cos, circRNA_sim)
    disease_similarity = disease_similarity_integrated(disease_GIP, disease_cos, disease_sim)


    return np.array(circRNA_similarity).astype(np.float32), np.array(disease_similarity).astype(np.float32)



