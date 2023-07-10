import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]

# Equalised learning rate - weighted, scaled convolution
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, gain = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size **2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialising conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self,x):
        return self.conv(x*self.scale)+ self.bias.view(1, self.bias.shape[0],1,1)
    

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.ε = 1e-8

    def forward(self,x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)+ self.ε)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super().__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn =PixelNorm()
        self.use_pn = use_pixelnorm

    def forward(self,x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x
    
class Generator(nn.Module):
    def __init__(self, noise_channel, in_channels, img_channels = 3): ## if using BW, change img channels to 1 
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(noise_channel, in_channels, 4,1,0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors) -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(WSConv2d(conv_out_channels,img_channels,kernel_size=1,stride=1,padding=0))

    def fade_in(self, alpha,upscaled, generated):
        return torch.tanh(alpha*generated +(1-alpha)* upscaled)
            
    def forward(self,x, alpha, steps):
        out = self.initial(x)

        if steps ==0:
            return self.initial_rgb(out)
        for step in range(steps):
            upscaled = F.interpolate(out,scale_factor=2,mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps -1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)
            


class Discriminator(nn.Module):
    def __init__(self, noise_channel, in_channels, img_channels=3):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors)-1,0,-1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors [i-1])
            self.prog_blocks.append(ConvBlock(conv_in_channels,conv_out_channels, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels,conv_in_channels,kernel_size=1,stride=1,padding=0))

        self.final_rgb = WSConv2d(img_channels, in_channels, kernel_size=1,stride=1, padding=0)
        self.rgb_layers.append(self.final_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
        self.final_block = nn.Sequential(
            WSConv2d(in_channels+1, in_channels, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4,stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels, 1, kernel_size=1, stride=1,padding=0 
            ),  # we use this instead of linear layer
        )

    def fade_in(self,alpha, downscaled, out):
        return alpha * out +(1-alpha)*downscaled

    def minibatch_std(self,x):
        batch_statistics = torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        return torch.cat([x,batch_statistics],dim=1) # to go from 512 -> 513.

    def forward(self,x, alpha, steps):
        current_step = len(self.prog_blocks) - steps  # gets us the last index
        
        out = self.leaky(self.rgb_layers[current_step](x))
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.leaky(self.rgb_layers[current_step +1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[current_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(current_step+1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)

        return self.final_block(out).view(out.shape[0],-1)
    

if __name__ == "__main__":
    NOISE_CHANNEL = 50
    IN_CHANNELS = 256
    gen = Generator(NOISE_CHANNEL, IN_CHANNELS, img_channels = 3)
    critic = Discriminator(NOISE_CHANNEL, IN_CHANNELS,img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 512, 1024]:
        num_steps = int(log2(img_size /4))
        x = torch.randn(1, NOISE_CHANNEL, 1,1,)
        z = gen(x,0.5, steps=num_steps)
        assert z.shape == (1,3,img_size,img_size)
        out = critic(z,alpha=0.5, steps=num_steps)
        assert out.shape == (1,1)
        print(f"Success! At img size: {img_size}")






        
