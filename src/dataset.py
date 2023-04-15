import torch
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

class PINNDataset:
    def __init__(self, Data):
        self.Data = Data
        
        self.R = torch.tensor(Data[:,0:2], dtype=torch.float32, requires_grad=True)
        self.UU = torch.tensor(Data[:,2:5], dtype=torch.float32, requires_grad=True)
        self.meanflow = torch.tensor(Data[:,5:], dtype=torch.float32, requires_grad=True)

    def __len__(self):
        return self.examples.shape[0]
    
    def __getitem__(self, str):
        #headers = ["He, "eta", "pi_p", "pi_m", "sig", "D", "M", "ubar", "dMdx", "dudx", "alp"]
        dict = {"R":self.R, "UU":self.UU, "meanflow":self.meanflow}
        return dict[str]
    


def get_orig_dataset():
    path = "/Users/ramtarun/Desktop/Cambridge/Indirect-Noise-in-Nozzles/Data/Data_PINN_ff0.01He0N551_subsonic.mat"
    data = scipy.io.loadmat(path)

    eta = data["eta"]
    pi_p = data["pi_p"]  
    pi_m = data["pi_m"]  
    sig = data["sig"]  
    D = data["D"]  
    M = data["M"]
    dMdx = data["dMdx"]    
    dudx = data["dudx"]  
    ubar = data["ubar"]  
    alp = data["alp"]  


    he = np.array([0.01])
    HE = np.tile(he, (1,551)) 
    #He = HE.flatten()[:,None]    #Initiailizing Helmontz Number
    

    # Rearrange Data
    R = np.zeros((551,2))
    R[:,0] = HE
    R[:,1] = eta

    UU = np.zeros((551,3))
    UU[:,0] = pi_p
    UU[:,1] = pi_m
    UU[:,2] = sig

    meanflow = np.zeros((551,7))
    meanflow[:,0] = D
    meanflow[:,1] = M
    meanflow[:,2] = ubar
    meanflow[:,3] = dMdx
    meanflow[:,4] = dudx
    meanflow[:,5] = alp
    meanflow[:,6] = eta

    N = R.shape[0]

    #NOISE_SCALE = 0.1
    #u += NOISE_SCALE * np.std(u) * np.random.randn(*u.shape)
    #v += NOISE_SCALE * np.std(v) * np.random.randn(*v.shape)

    # Split Dataset
    R_train, R_test, UU_train, UU_test = train_test_split(R, UU, train_size=0.85, shuffle=False)
    meanflow_train, meanflow_test = train_test_split(meanflow, train_size=0.85, shuffle=False)

    train_data = np.hstack((R_train, UU_train, meanflow_train))
    test_data = np.hstack((R_test, UU_test, meanflow_test))
    train_data = PINNDataset(train_data)
    test_data = PINNDataset(test_data)

    return train_data, test_data