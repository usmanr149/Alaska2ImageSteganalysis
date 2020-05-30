import numpy as np
from collections import Counter

def SPAM(img, i_prime=0, j_prime=3, T=4):
    T+=1
    diff_img = []
    for i in range(0, img.shape[0], 1):
        for j in range(0, img.shape[1], 1):
            if j+j_prime<img.shape[1] and i+i_prime<img.shape[0]:
                diff = img[i][j] - img[i+i_prime][j+j_prime]
                if abs(diff) < T:
                    diff_img.append(diff)
    return np.array(diff_img)

def SPAM_Dh(img, j_prime=3, T=3):
    #Dh
    diff = (img[:,:img.shape[1]-j_prime] - img[:, j_prime:]).ravel()
    diff = diff[np.logical_and(diff>=-T, diff<=T)]
    return diff

def SPAM_Dv(img, i_prime=3, T=3):
    #Dv
    diff = (img[:img.shape[0]-i_prime] - img[i_prime:]).ravel()
    diff = diff[np.logical_and(diff>=-T, diff<=T)]
    return diff

def SPAM_Dd(img, i_prime=3, j_prime=3, T=3):
    # get width
    w = img.shape[1]

    img_rav = img.ravel()

    a = img_rav[:img_rav.shape[0]-(j_prime*w+i_prime)]
    b = img_rav[j_prime*w+i_prime:]

    for i in range(i_prime):
        try:
            del_ = np.append(del_, np.arange(w-i-1,img_rav.shape[0]-(j_prime*w+i_prime), w))
        except:
            del_ = np.arange(w-i-1,img_rav.shape[0]-(j_prime*w+i_prime), w)
    a = np.delete(a, del_, axis=0)
    b = np.delete(b, del_, axis=0)

    diff = a - b
    diff = diff[np.logical_and(diff>=-T, diff<=T)]

    return diff

def SPAM_Dh_count(img, j_prime, T):
    #Dh
#     img = img.astype(int)
    diff =  img[:, j_prime:] - img[:,:img.shape[1]-j_prime]
    diff = diff + T

#     M = [[0]*(2*T+1) for _ in range(2*T+1)]
    M = np.zeros((2*T+1,2*T+1))
    for row in diff:
        for (i,j), c in Counter(zip(row, row[1:])).items():
            if i >= 0 and i<=2*T and j>=0 and j<=2*T:
                M[i][j] += c

    return M

def SPAM_Dv_count(img, i_prime, T):
    #Dh
#     img = img.astype(int)
    diff =  img[i_prime:] - img[:img.shape[0]-i_prime]
    diff = diff + T

    # M = [[0]*(2*T+1) for _ in range(2*T+1)]
    M = np.zeros((2*T+1,2*T+1))
    for row in diff.T:
        for (i,j), c in Counter(zip(row, row[1:])).items():
            if i >= 0 and i<=2*T and j>=0 and j<=2*T:
                M[i][j] += c

    return M

def SPAM_Dd_count(img, i_prime=3, j_prime=3, T=3):
    #Dh
#     img = img.astype(int)
    diff =  img[i_prime:, j_prime:] - img[:img.shape[0]-i_prime, : img.shape[1]-j_prime]
    diff = diff + T
#     print(diff.shape)

    M = np.zeros((2*T+1,2*T+1))

    for k in range(-diff.shape[1] + 2 , diff.shape[1] - 1):
        diag = np.diag(diff, k)
        for (i,j), c in Counter(zip(diag, diag[1:])).items():
            if i >= 0 and i<=2*T and j>=0 and j<=2*T:
                M[i][j] += c

    return M

def count_to_transition(M):

    M = M/M.sum(axis=1)[:,None]
    M[np.isnan(M)] = 0
    return M


def transition_matrix(transitions, T=3):
    n = 2*T+1 #number of states

    M = np.zeros((n,n))

    for (i,j), c in Counter(zip(transitions, transitions[1:])).items():
        M[i][j] = c

    M = M/M.sum(axis=1)[:,None]
    M[np.isnan(M)] = 0
    return M
