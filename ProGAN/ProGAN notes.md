ProGAN notes

Progressive growing of GANs for improved quality, stability and variation.
Karras, Aila, Laine, Lehtinen 2018.

Brand new training method for GANs.

1) start from low resolution and progressive growing.
4x4 -> 8x8 -> 16x16 -> ...

2) Mini batch std

3) Pixel Norm

4) Equalised learning rate.


Generator and discriminator are mirrors of one another 4x4 -> 16x16 :: 16x16 -> 4x4

mean of the standard deviation over pixels and feature channels.

blends and fades in new layers smoothly using a scalar alpha 0 -> 1 

Adam and RMSProp are well suited to dynamic gradients, but in this paper, they use He initializer to allow more stable gradient descent strat.
(dynamic range is low with He initializer which is beneficial for non-stable GANs).


Pixel norm used to disallow the scenario where the magnitudes in the generator and discriminator sprial out of control. 
They normalize the feature vector in each pixel to unit length in the generator after each conv layer.
(batchnorm has inherent problems, dependence on batch size, impact of model parrallelism, RNN problems, memory ocnsumption, generalization)