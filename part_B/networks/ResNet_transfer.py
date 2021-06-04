import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class TransferResNet(nn.Module):
    def __init__(self, fc_model_shape):
        super(TransferResNet, self).__init__()
        self.model = models.resnet18(True)
        print(self.model)

        for param in self.model.parameters():
            param.requires_grad = False

        n_inputs = self.model.fc.in_features
        # self.model = nn.Sequential(*list(self.model.children())[:-1])

        # self.model.fc = 
        layers = []
        layers.append(torch.nn.Linear(n_inputs,fc_model_shape[0]).float())
        for i in range(0, len(fc_model_shape)-1):
            layer = torch.nn.Linear(fc_model_shape[i], 
                                    fc_model_shape[i+1]).float()
            layers.append(torch.nn.ReLU())
            layers.append(layer)
        
        layers = torch.nn.Sequential(*layers)
        self.model.fc = layers
        
        print(self.model)
        #print("trainable params: " + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
#


    def forward(self,x):
        out = self.model(x)

        out = out.reshape(-1,1,240,320)
        m = nn.Upsample(scale_factor=2, mode='bilinear')

        out = m(out)

        out = torch.squeeze(out)
#        print(out.shape)
        return out


# def main():
#     print("ELO GOVENER")
#     model = TransferResNet()
#     print(model)

# if __name__ == "__main__":
#     main()
