import torch
import torch.nn as nn


class VGG_New(nn.Module):
  def __init__(self,num_classes = 10):
    super(VGG_New, self).__init__()

    self.featchers = nn.Sequential(
        nn.Conv2d(in_channels = 3 , out_channels = 64 , kernel_size = 3 , padding=1),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = 64 , out_channels = 64 , kernel_size = 3 , padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(kernel_size=4, stride=2),

        nn.Conv2d(in_channels = 64 , out_channels = 256 , kernel_size = 3 , padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(in_channels = 256 , out_channels = 256,kernel_size = 4 , padding = 1 , stride=2),

        # nn.ReLU(inplace=True),    ###1

        nn.Conv2d(in_channels = 256 , out_channels = 512 , kernel_size=3 , padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels = 512 , out_channels = 512 , kernel_size=3 , padding=1),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=4, stride=2),
        
        nn.ConvTranspose2d(in_channels = 512 , out_channels = 512,kernel_size = 4 , padding = 1 , stride=2),
        
        # nn.ReLU(inplace=True), ###2
        
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),


        nn.MaxPool2d(kernel_size=4, stride=2),  # 109 * 109 * 512


        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),


        ########################################################################################
        # nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # nn.ReLU(inplace=True),


        nn.MaxPool2d(kernel_size=4, stride=2),   #53 * 53 * 512
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.ConvTranspose2d(in_channels = 512 , out_channels = 512,kernel_size = 4 , padding = 1 , stride=2),

        nn.ReLU(inplace=True),
        
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=4, stride=2),


        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 128, kernel_size=3, padding=1),
  
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),



    )
    self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

    self.classifier = nn.Sequential(
            nn.Linear(64*8*8, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, num_classes)
        )

  def forward(self, x):
        x = self.featchers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x