import torch
from torch import nn, autograd, Tensor
from torch.nn import functional as F

from PDE import RHS_f

def calc_grad(y, x) -> Tensor:
    grad = autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


# Neural Network for Predicting the pi+, pi-, and si.

class NN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers 

        'activation function'
        self.activation = nn.Tanh()
    
        'Initialize neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        self.dropout = nn.Dropout(0.1)
    
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)


    def forward(self, X):
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X) 
        a = X.type(torch.float32)
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
            a = self.dropout(a)
        a = self.linears[-1](a)
        return a
    


# Physics informed Neural Network 
f = torch.empty(1)

class PINN(nn.Module):

    """
    `forward`: returns a tensor of shape (D, 3), where D is the number of
    data points, and the 2nd dim. is the predicted values of pi_p, pi_m, sig.
    """

    def __init__(self, X,  uu, meanflow, layers):
        super().__init__()
        self.loss_function = nn.MSELoss(reduction ='mean')

        'Initialize our new parameters i.e.f (Inverse problem)' 
        self.f = torch.tensor([f], requires_grad=True).type(torch.float32)
        nn.init.zeros_(self.f)
        'Register f to optimize'
        self.f = nn.Parameter(self.f)
        'Initialize iterator'
        self.iter = 0
        'Call our DNN'
        self.dnn = NN(X,layers)
        'Register our new parameter'
        self.dnn.register_parameter('f', self.f)  
        
        self.meanflow = meanflow
        self.UU = uu

        self.X = X
        self.Eta = X[:, 1]

    def loss_data(self, X):
        loss_u = self.loss_function(self.dnn(X), self.UU)
        return loss_u
    
    def loss_residual(self, X):
        f = self.f
        g = torch.clone(X)
        R = self.dnn(g)
        GR = RHS_f(R, self.meanflow, f)
        dr_dn = calc_grad(R,g)
        pde = dr_dn[:, 1:] + GR
        return torch.mean(pde**2)
    
    def Loss(self, X):
        loss_data = self.loss_data(X)
        loss_residual = self.loss_residual(X)
        return loss_data + loss_residual
    

    'callable for optimizer'                                       
    def closure(self):
        
        optimizer.zero_grad()
        
        loss = self.Loss(self.X)
        
        loss.backward()
                
        self.iter += 1

        
        
        if self.iter % 200 == 0:
            print('f_real = [0.01], f_PINN = [%.5f], Loss = [%.5f] ' %(self.f.item(), loss))
            
        return loss