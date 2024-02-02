import torch
import torch.nn as nn

class MMDLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def gram_RBF(self, x, y, sigma):
        gram=[]
        for x_ in x:
            for y_ in y:
                gram.append(torch.exp(-torch.sum((x_-y_)**2)/sigma))
        return torch.stack(gram).view(x.shape[0],y.shape[0])
    

    def forward(self, input, target,sigma):
        # Implement your custom loss calculation here
        XX = self.gram_RBF(input,input,sigma)
        YY = self.gram_RBF(target,target,sigma)
        XY = self.gram_RBF(input,target,sigma)
        loss = torch.mean(XX) + torch.mean(YY) - 2*torch.mean(XY)
        return loss