import torch
import torch.nn as nn
from utils import orthogonal_init
from my_util import xavier_uniform_init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
  """
      Framework for encoder
  """
  def __init__(self):
    super().__init__()

  def forward(self):
    raise NotImplementedError

#region Nature
#Called NatureModel in https://github.com/joonleesky/train-procgen-pytorch/blob/master/common/model.py
class NatureEncoder(Encoder):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), 
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), 
        nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)
#endregion

#region Impala
class ImpalaEncoder(Encoder):
  """
    Taken directly from:
    https://github.com/joonleesky/train-procgen-pytorch/blob/master/common/model.py

  """
  def __init__(self,
                in_channels,
                feature_dim):
    super().__init__()

    self.output_dim = feature_dim

    self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
    self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
    self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
    self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=self.output_dim)

    self.apply(xavier_uniform_init)

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = nn.ReLU()(x)
    x = Flatten()(x)
    x = self.fc(x)
    x = nn.ReLU()(x)
    return x

    
class ResidualBlock(nn.Module):
  """
    Taken directly from:
    https://github.com/joonleesky/train-procgen-pytorch/blob/master/common/model.py

  """
  def __init__(self,
                  in_channels):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    out = nn.ReLU()(x)
    out = self.conv1(out)
    out = nn.ReLU()(out)
    out = self.conv2(out)
    return out + x

class ImpalaBlock(nn.Module):
  """
    Taken directly from:
    https://github.com/joonleesky/train-procgen-pytorch/blob/master/common/model.py

  """
  def __init__(self, in_channels, out_channels):
    super(ImpalaBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
    self.res1 = ResidualBlock(out_channels)
    self.res2 = ResidualBlock(out_channels)

  def forward(self, x):
    x = self.conv(x)
    x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
    x = self.res1(x)
    x = self.res2(x)
    return x
  #endregion