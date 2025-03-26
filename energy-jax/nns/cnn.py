"""CNNs."""

from typing import Optional, List, Callable, Sequence, Any, Union, Type, Tuple, Dict
from functools import partial
import jax
from jax import numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray
import equinox as eqx  # type: ignore[import]


class MnistCNN(eqx.Module):
    """
    An energy function based on a standard MNIST CNN architecture.

    Attributes:
        - layers: the list of sequential layers
    """

    layers: list

    def __init__(
        self, key: PRNGKeyArray, outputs: int = 1, hidden_features: int = 32
    ) -> None:
        """
        Initialize the neural network.

        Args:
            - key: the random key to use for initialization
            - output: the number of outputs of the neural network
            - hidden_features: the number of hidden features to have the final CNN layer output,
                this must be a even multiple of 2.

        Returns:
            - None
        """
        if not isinstance(hidden_features, int):
            raise TypeError("hidden_features must be an integer")
        chid1 = hidden_features // 2
        chid2 = hidden_features
        chid3 = hidden_features * 2

        keys = jax.random.split(key, 6)

        self.layers = [
            eqx.nn.Conv2d(1, chid1, kernel_size=5, stride=2, padding=4, key=keys[0]),
            jax.nn.swish,
            eqx.nn.Conv2d(
                chid1, chid2, kernel_size=3, stride=2, padding=1, key=keys[1]
            ),
            jax.nn.swish,
            eqx.nn.Conv2d(
                chid2, chid3, kernel_size=3, stride=2, padding=1, key=keys[2]
            ),
            jax.nn.swish,
            eqx.nn.Conv2d(
                chid3, chid3, kernel_size=3, stride=2, padding=1, key=keys[3]
            ),
            jax.nn.swish,
            lambda x: x.flatten(),
            eqx.nn.Linear(chid3 * 4, chid3, key=keys[4]),
            jax.nn.swish,
            eqx.nn.Linear(chid3, outputs, key=keys[5]),
        ]

    def __call__(self, x: Float[Array, "28 28"]) -> Float[Array, ""]:
        """
        Forward pass of the neural network.

        Args:
            - x: a 28x28 array

        Returns:
            - the energy function of this input
        """
        x = jnp.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28) where 1 is # channels
        for layer in self.layers:
            x = layer(x)
        return jnp.squeeze(x)


def _convnxn(
    in_planes: int,
    out_planes: int,
    kernel_size: int,
    stride: Union[int, Sequence[int]] = 1,
    groups: int = 1,
    dilation: Union[int, Sequence[int]] = 1,
    key: Optional[PRNGKeyArray] = None,
) -> eqx.nn.Conv2d:
    """
    N x N convolution with padding.

    Args:
        - in_planes: number of input channels
        - out_planes: number of output channels
        - kernel_size: the size of the kernel
        - stride: the stride of the convolution
        - groups: the number of input channel groups
        - dilation: the dilation of the convolution
        - key: the random key to use for initialization

    Returns:
        - an Conv2d layer
    """
    if key is None:
        raise ValueError("key cannot be None")
    return eqx.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation,
        groups=groups,
        use_bias=False,
        dilation=dilation,
        key=key,
    )


def _convnxn_no_pad(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 1,
    stride: Union[int, Sequence[int]] = 1,
    key: Optional[PRNGKeyArray] = None,
    use_bias: bool = False,
) -> eqx.nn.Conv2d:
    """
    N x N convolution without padding.

    Args:
        - in_planes: number of input channels
        - out_planes: number of output channels
        - kernel_size: the size of the kernel
        - stride: the stride of the convolution
        - key: the random key to use for initialization
        - use_bias: whether or not to use a bias

    Returns:
        - an Conv2d layer
    """
    if key is None:
        raise ValueError("key cannot be None")
    return eqx.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        use_bias=use_bias,
        key=key,
    )


class EqxvisionResidualBlock(eqx.Module):
    """
    Single residual block.

    Adapted (along with resnet) from eqxvision.

    Computes f(x) = x + g(x).

    eqxvision license is available at https://github.com/paganpasta/eqxvision/blob/main/LICENSE.md

    Attributes:
        - conv1: the first convolution operation
        - norm1: the first normalization operation
        - relu: the ReLU activiation function
        - conv2: the second convolution
        - norm2: the second normalization operation
        - downsample: the operation to convert x to the same shape as g(x)
        - stride: the stride of the convolutions
    """

    conv1: eqx.nn.Conv2d
    norm1: eqx.Module
    relu: Callable
    conv2: eqx.nn.Conv2d
    norm2: eqx.Module
    downsample: eqx.Module
    stride: int

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[eqx.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> None:
        """
        Initialize convolutions and default norm layers.

        Args:
            - inplanes: the input channels
            - planes: output channels
            - stride: the stride of the convolution
            - downsample: the downsampling operation
            - groups: the groups for the convolution
            - dilation: the dilation of the convolution operations
            - norm_layer: the normalization layer to use
            - key: the random key to use for initialization

        Returns:
            - None
        """
        if key is None:
            raise ValueError("key must be specified")
        if norm_layer is not None:
            self.norm1 = norm_layer(planes)
            self.norm2 = norm_layer(planes)
        else:
            self.norm1 = eqx.nn.Identity()
            self.norm2 = eqx.nn.Identity()
        if groups != 1:
            raise ValueError("Groups must equal 1")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        keys = jax.random.split(key, 2)
        self.conv1 = _convnxn(inplanes, planes, 3, stride, key=keys[0])
        self.relu = jax.nn.relu
        self.conv2 = _convnxn(planes, planes, 3, key=keys[1])
        if downsample:
            self.downsample = downsample
        else:
            self.downsample = eqx.nn.Identity()
        self.stride = stride

    def __call__(
        self,
        x: Float[Array, "in_channels dim1 dim2"],
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "out_channels out_dim1 out_dim2"]:
        """
        Forward pass of residual block.

        Args:
            - x: the input tensor
            - key: unused key (but accepted as input to match eqx typing)

        Returns:
            - the result of the forward pass
        """
        if not (
            callable(self.norm1) and callable(self.norm2) and callable(self.downsample)
        ):
            raise ValueError("norm1, norm2, downsample must be callable")
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(eqx.Module):
    """
    An adaptation of eqxvision's ResNet which adapts torchvision.models.resnet.

    Based on https://arxiv.org/abs/1512.03385.

    Attributes:
        - inplanes: the input channels
        - dilation: the dilation of the convolutions
        - groups: the groups of channels
        - conv1: the first convolution operation
        - norm1: the first normalization operation
        - relu: the ReLU activation function
        - maxpool: a maximum pooling operation
        - layer1: the first residual block
        - layer2: the second residual block
        - layer3: the third residual block
        - layer4: the fourth residual block
        - avgpool: an average pooling layer
        - fc: the fully connected output layer
    """

    inplanes: int
    dilation: int
    groups: int
    conv1: eqx.nn.Conv2d
    norm1: eqx.Module
    relu: Callable
    maxpool: eqx.nn.MaxPool2d
    layer1: eqx.nn.Sequential
    layer2: eqx.nn.Sequential
    layer3: eqx.nn.Sequential
    layer4: eqx.nn.Sequential
    avgpool: eqx.nn.AdaptiveAvgPool2d
    fullyconnected: eqx.nn.Linear

    def __init__(
        self,
        block: Type[EqxvisionResidualBlock],
        layers: List[int],
        num_classes: int = 1,
        groups: int = 1,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> None:
        """
        Create residual blocks and initialize member variables.

        Args:
            - block: a basic block for constructing the network
            - layers: A list containing number of blocks at different levels
            - num_classes: Number of classes in the classification task.
                            Also controls the final output shape (num_classes,). Defaults to 1.
            - groups: Number of groups to form along the feature depth. Defaults to 1.
            - replace_stride_with_dilation: Replacing 2x2 strides with dilated
                convolution. Defaults to None.
            - norm_layer: Normalisation to be applied on the inputs. Defaults to None.
            - key: A jax.random.PRNGKey used to provide randomness for parameter initialisation.

        Returns:
            - None

        Raises:
            - ValueError: If replace_stride_with_convolution is not None or a 3-tuple
        """
        if key is None:
            raise TypeError("key cannot be None.")
        keys = jax.random.split(key, 6)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.conv1 = eqx.nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=keys[0],
        )
        if norm_layer is None:
            norm_layer = eqx.nn.Identity
        self.norm1 = norm_layer(self.inplanes)
        self.relu = jax.nn.relu
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer, key=keys[1])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[0],
            key=keys[2],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[1],
            key=keys[3],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[2],
            key=keys[4],
        )
        self.avgpool = eqx.nn.AdaptiveAvgPool2d((1, 1))
        self.fullyconnected = eqx.nn.Linear(512, num_classes, key=keys[5])

    def _make_layer(
        self,
        block: Type[EqxvisionResidualBlock],
        planes: int,
        blocks: int,
        norm_layer: Callable,
        stride: int = 1,
        dilate: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> eqx.nn.Sequential:
        """
        Private constructor for residual block.

        Args:
            - block: the residual block to use
            - planes: the input channels
            - blocks: the number of residual blocks to make within this layer
            - norm_layer: the normalization layer to use
            - stride: the stride of the convolutions
            - dilate: whether to dilate the convolutions
            - key: the random key to use for initialization

        Returns:
            - a sequential module of the blocks layer
        """
        if key is None:
            raise ValueError("key must be specified")
        keys = jax.random.split(key, blocks + 1)
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = eqx.nn.Sequential(
                [
                    _convnxn_no_pad(self.inplanes, planes, 1, stride, key=keys[0]),
                    norm_layer(planes),
                ]
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                previous_dilation,
                norm_layer,
                key=keys[1],
            )
        )
        self.inplanes = planes
        for block_idx in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    key=keys[block_idx + 1],
                )
            )

        return eqx.nn.Sequential(layers)

    def __call__(
        self,
        x: Float[Array, "channels in_dim1 in_dim2"],
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "num_classes"]:
        """
        Compute a forward pass of the full ResNet model.

        Args:
            - x: the input to compute the forward pass on.
            - key: can be utilised by few layers such as Dropout or DropPath, also aligns with
                equinox typing more broadly.

        Returns:
            - the result of the forward pass
        """
        if not (callable(self.norm1)):
            raise ValueError("norm1 must be callable")
        if key is None:
            keys = [None] * 6
        else:
            keys = jax.random.split(key, 6)  # type: ignore
        x_output = self.conv1(x, key=keys[0])
        x_output = self.norm1(x_output)
        x_output = self.relu(x_output)
        x_output = self.maxpool(x_output)

        x_output = self.layer1(x_output, key=keys[1])
        x_output = self.layer2(x_output, key=keys[2])
        x_output = self.layer3(x_output, key=keys[3])
        x_output = self.layer4(x_output, key=keys[4])

        x_output = self.avgpool(x_output)
        x_output = jnp.ravel(x_output)
        x_output = self.fullyconnected(x_output, key=keys[5])

        return x_output


def _resnet(
    block: Type[EqxvisionResidualBlock], layers: List[int], **kwargs: Any
) -> ResNet:
    """
    Create a resnet model.

    Args:
        - block: the basic block to use
        - layers: the blocks per layer
        - **kwargs: the keyword arguments for the rest of the resnet

    Returns:
        - the resulting resnet model
    """
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs: Any) -> ResNet:
    """
    Make a ResNet 18 model.

    Args:
        - **kwargs: the ResNet arguments

    Returns:
        - the resnet model
    """
    model = _resnet(EqxvisionResidualBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs: Any) -> ResNet:
    """
    Make a ResNet-34 model.

    Args:
        - **kwargs: the ResNet arguments

    Returns:
        - the resnet model
    """
    model = _resnet(EqxvisionResidualBlock, [3, 4, 6, 3], **kwargs)
    return model


class _UNetResBlock(eqx.Module):
    """
    Residual block for a UNet.

    This UNet implementation is adapted from:
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

    Attributes:
        - conv1: the first convolution
        - act: the activation function
        - conv2: the second convolution
        - downsample: the downsampling operation used for the residual connection, in
            this case it is a 1x1 convolution, same as in the original ResNet paper
    """

    conv1: eqx.nn.Conv2d
    act: Callable
    conv2: eqx.nn.Conv2d
    downsample: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        key: PRNGKeyArray,
        hid_channels: Optional[int] = None,
    ) -> None:
        """
        Initialize the block.

        Args:
            - in_channels: the input channels
            - out_channels: the output channels
            - key: the random key to use for initialization
            - hid_channels: the channels to use for the intermediate step

        Returns:
            - None
        """
        if hid_channels is None:
            hid_channels = out_channels
        subkey1, subkey2, subkey3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(
            in_channels, hid_channels, kernel_size=3, padding=1, key=subkey1
        )
        self.act = lambda x: jax.nn.leaky_relu(x, 0.2)
        self.conv2 = eqx.nn.Conv2d(
            hid_channels, out_channels, kernel_size=3, padding=1, key=subkey2
        )
        self.downsample = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, key=subkey3
        )

    def __call__(
        self, x: Float[Array, "in_channels dim dim"]
    ) -> Float[Array, "out_channels dim dim"]:
        """
        Compute a forward pass of the block.

        Args:
            - x: the input array

        Returns:
            - the array output of the block
        """
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + self.downsample(x)
        out = self.act(out)
        return out


class _UNetDown(eqx.Module):
    """
    Downsample operation of a UNet.

    Attributes:
        - maxpool: the maxpooling operation
        - block: the resnet block
    """

    maxpool: eqx.nn.MaxPool2d
    block: _UNetResBlock

    def __init__(self, in_channels: int, out_channels: int, key: PRNGKeyArray) -> None:
        """
        Initialize the downsampling operation.

        Args:
            - in_channels: the input channels
            - out_channels: the output channels
            - key: the random key to use for initialization

        Returns:
            - None
        """
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = _UNetResBlock(in_channels, out_channels, key=key)

    def __call__(
        self, x: Float[Array, "in_channels dim dim"]
    ) -> Float[Array, "out_channels dim/2 dim/2"]:
        """
        Compute a forward pass of the downsample.

        Args:
            - x: the input array

        Returns:
            - the array output of the block
        """
        x_output = self.maxpool(x)
        x_output = self.block(x_output)
        return x_output


class _UNetUp(eqx.Module):
    """
    Upsample operation of a UNet.

    Attributes:
        - up: the convolutional transpose operation
        - block: the resnet block
    """

    upsampling_layer: eqx.nn.ConvTranspose2d
    block: _UNetResBlock

    def __init__(self, in_channels: int, out_channels: int, key: PRNGKeyArray) -> None:
        """
        Initialize the upsample operation.

        Args:
            - in_channels: the input channels
            - out_channels: the output channels
            - key: the random key to use for initialization

        Returns:
            - None
        """
        subkey1, subkey2 = jax.random.split(key)
        self.upsampling_layer = eqx.nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=1, key=subkey1
        )
        self.block = _UNetResBlock(in_channels, out_channels, key=subkey2)

    def __call__(
        self,
        x1_prev: Float[Array, "in_channel1 dim1 dim1"],
        x2_down: Float[Array, "in_channel2 dim2 dim2"],
    ) -> Float[Array, "in_channel2 dim2 dim2"]:
        """
        Compute a forward pass of the upsample.

        Args:
            - x1_prev: the input from the previous layer
            - x2_down: the input from the respective downsampled layer (this is
                what flows information through and differentiates it from
                something like an autoencoder)

        Returns:
            - the array output of the block
        """
        x1_prev = self.upsampling_layer(x1_prev)
        diffY_dim = x2_down.shape[1] - x1_prev.shape[1]
        diffX_dim = x2_down.shape[2] - x1_prev.shape[2]
        x1_prev = jnp.pad(
            x1_prev,
            [
                (0, 0),
                (diffX_dim // 2, diffX_dim - diffX_dim // 2),
                (diffY_dim // 2, diffY_dim - diffY_dim // 2),
            ],
        )  # 0 pad x1 to match x2's shape
        x_concat = jnp.concatenate((x2_down, x1_prev), axis=0)
        return self.block(x_concat)


class UNet(eqx.Module):
    """
    UNet architecture.

    The UNet architecture, first introduced by https://arxiv.org/abs/1505.04597, for image
    segmentation, is a image to image network. It has two paths, a contracting path which consists
    of repeated maxpool based downsampling, and an expanding path, which consists of transpose
    convolutions to upscale the image. This downscale and upscaling yield a U like architecture
    (see fig 1 of the paper). Note that this is distinct from something like an autoencoder, because
    there isn't an information bottleneck. The information from each part of the contracting path is
    passed to the expanding path.

    Attributes:
        - inc: the initial convolution residual block
        - downs: the downsampling operations
        - ups: the upsampling operations
        - conv_out: the final convolutional output
    """

    inc: _UNetResBlock
    downs: List[_UNetDown]
    ups: List[_UNetUp]
    conv_out: eqx.nn.Conv2d

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        key: PRNGKeyArray,
        depth: int = 2,
        max_val: int = 128,
    ) -> None:
        """
        Initialize the UNet network.

        Assumes that each layers number of channels increases by a factor of two and the
        size of the image shrinks by a factor of 2. This means that your input image size, and
        the depth/max_val of the UNet must be compatible. This is not checked for explicitly.

        Args:
            - in_channels: the input channels
            - out_channels: the output channels
            - key: the random key to use for initialization
            - depth: the number of down/upsampling layers
            - max_val: the maximum number of channels

        Returns:
            - None
        """
        subkeys = jax.random.split(key, depth * 2 + 2)
        vals = [max_val // (2**i) for i in range(depth + 1)][::-1]
        self.inc = _UNetResBlock(n_channels, vals[0], key=subkeys[0])
        self.downs = [
            _UNetDown(vals[i - 1], vals[i], subkeys[i]) for i in range(1, depth + 1)
        ]
        self.ups = [
            _UNetUp(vals[i], vals[i - 1], subkeys[i + depth])
            for i in range(depth, 0, -1)
        ]
        self.conv_out = eqx.nn.Conv2d(
            vals[0], n_classes, kernel_size=1, key=subkeys[-1]
        )

    def __call__(
        self, x: Float[Array, "n_channels dim dim"]
    ) -> Float[Array, "n_classes dim dim"]:
        """
        Forward pass of the UNet.

        Args:
            - x: the original input image

        Returns:
            - an array with the same image shape as the input, but the defined number of channels,
                the result of the forward pass of the UNet
        """
        xs_inc = [self.inc(x)]
        for down in self.downs:
            xs_inc.append(down(xs_inc[-1]))
        x = xs_inc[-1]
        for i, up_layer in enumerate(self.ups):
            x = up_layer(x, xs_inc[-(i + 2)])
        logits = self.conv_out(x)
        return logits


class InstanceNorm2dPlus(eqx.Module):
    """
    Variant of Instance Normalization used in the NCSN works.

    This is used in NCSN2 (https://arxiv.org/abs/2006.09011) and is the equations
    in A.1 of https://arxiv.org/abs/1907.05600 but with number of classes set to 1.

    Adapted from: https://github.com/ermongroup/ncsnv2/blob/master/models/normalization.py#L150

    Attributes:
        - num_features: the number of features (channels)
        - instance_norm: the standard instance norm (note that instance norm is just group
            norm with number of groups = number of features)
        - alpha: learned parameter which reintroduces mean information (to reduce colorshift
            in generated images)
        - gamma: the gamma parameter in instance norm
        - bias: the optional bias parameters of the layer
    """

    num_features: int
    instance_norm: eqx.nn.GroupNorm
    alpha: jnp.ndarray
    gamma: jnp.ndarray
    bias: Optional[jnp.ndarray] = None

    def __init__(
        self, num_features: int, key: PRNGKeyArray, use_bias: bool = True
    ) -> None:
        """
        Randomly initialize the member attributes.

        Args:
            - num_features: the number of features
            - key: the random key to use for initialization
            - use_bias: whether or not to use a bias

        Returns:
            - None
        """
        self.num_features = num_features
        self.instance_norm = eqx.nn.GroupNorm(num_features, channelwise_affine=False)
        subkey1, subkey2 = jax.random.split(key, 2)
        self.alpha = jax.random.normal(subkey1, (num_features,)) * 0.02 + 1
        self.gamma = jax.random.normal(subkey2, (num_features,)) * 0.02 + 1
        if use_bias:
            self.bias = jnp.zeros((num_features,))

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        """
        Compute the forward pass of the instance norm.

        This computes gamma * (instance_norm + means * a) + b, but
        the paper says gamma * instance_norm + means * a + b. The official code
        release does the former, which we follow.

        Args:
            - x: the input array

        Returns:
            - the result of the forward pass
        """
        means = jnp.mean(x, axis=(1, 2))
        m = jnp.mean(means)
        v = jnp.var(means)
        means = (means - m) / (jnp.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias is not None:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = jnp.reshape(self.gamma, (self.num_features, 1, 1)) * h + jnp.reshape(
                self.bias, (self.num_features, 1, 1)
            )
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = jnp.reshape(self.gamma, (self.num_features, 1, 1)) * h
        return out


def _dilated_conv3x3(
    in_planes: int, out_planes: int, key: PRNGKeyArray, dilation: int, bias: bool = True
) -> eqx.nn.Conv2d:
    """
    Dilated 3x3 convolution.

    Dilated convolution, also known as atrous convolution, is a variant of the
    standard convolution operation, where the filter is applied to the input
    with gaps. This is controlled by the dilation rate. For example, a 3x3 filter
    with a dilation rate of 2 will have the same field of view as a 5x5 filter,
    but with only 9 parameters. This allows the network to have larger receptive
    fields with fewer parameters.

    Args:
        - in_planes: the number of input channels
        - out_planes: the number of output channels
        - key: the random key to use for initialization
        - dilation: the scale of the dilation
        - bias: whether to use bias or not

    Returns:
        - an equinox 2d convolution
    """
    if dilation is None:
        padding = 0
        dilation = 1
    else:
        padding = dilation
    return eqx.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=padding,
        dilation=dilation,
        use_bias=bias,
        key=key,
    )


def _conv3x3(
    in_planes: int,
    out_planes: int,
    key: PRNGKeyArray,
    stride: int = 1,
    bias: bool = True,
) -> eqx.nn.Conv2d:
    """
    3x3 convolution.

    Args:
        - in_planes: the number of input channels
        - out_planes: the number of output channels
        - key: the random key to use for initialization
        - stride: the scale of the stride
        - bias: whether to use bias or not

    Returns:
        - an equinox 2d convolution
    """
    return eqx.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        use_bias=bias,
        key=key,
    )


class _ConvMeanPool(eqx.Module):
    """
    A module that Perform a 2D convolution followed by a mean pooling operation.

    This class _ConvMeanPool is used to apply a 2D convolution on the input tensor followed
    by a mean pooling operation. The convolution operation is performed using the eqx.nn.Conv2d
    layer. The mean pooling operation is performed by taking the average of the convolved
    output at different strides.

    Adapted from: https://github.com/ermongroup/ncsnv2/blob/master/models/layers.py#L291

    Attributes:
        - conv: A 2D convolution layer.
        - pad: A flag to determine whether to adjust padding or not.
        - padding: A tuple specifying the padding dimensions.
    """

    conv: eqx.nn.Conv2d
    pad: bool
    padding: Tuple[int, int, int, int]

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        key: PRNGKeyArray,
        kernel_size: int = 3,
        bias: bool = True,
        adjust_padding: bool = False,
    ) -> None:
        """
        Initialize the _ConvMeanPool module with the given parameters.

        Args:
            - input_dim: The number of input channels.
            - output_dim: The number of output channels.
            - key: The key for the random number generator.
            - kernel_size: The size of the kernel. Defaults to 3.
            - bias: If True, adds a learnable bias to the output. Defaults to True.
            - adjust_padding: If True, adjusts the padding of the input. Defaults to False.

        Returns:
            - None
        """
        self.conv = eqx.nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            use_bias=bias,
            key=key,
        )
        self.pad = adjust_padding
        self.padding = (1, 0, 1, 0)

    def __call__(
        self, x: Float[Array, "input_dim h w"]
    ) -> Float[Array, "output_dim h/2 w/2"]:
        """
        Perform the forward pass of the _ConvMeanPool module.

        Args:
            - x: The input tensor.

        Returns:
            - output: The output tensor after the convolution and mean pooling operations.
        """
        if self.pad:
            pad_width = (
                (0, 0),
                (self.padding[2], self.padding[3]),
                (self.padding[0], self.padding[1]),
            )
            x = jnp.pad(x, pad_width, mode="constant", constant_values=0)
        output = self.conv(x)
        output = jnp.array(
            sum(
                [
                    output[:, ::2, ::2],
                    output[:, 1::2, ::2],
                    output[:, ::2, 1::2],
                    output[:, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return output


class _NCSNResidualBlock(eqx.Module):
    """
    A module that implements a residual block used in the NCSN model.

    This class _NCSNResidualBlock is used to apply a sequence of operations including
    normalization, non-linearity, and convolution on the input tensor. The output
    is then added to the original input (or its transformed version), forming a
    residual connection.

    Adapted from: https://github.com/ermongroup/ncsnv2/blob/master/models/layers.py#L401

    Attributes:
        - non_linearity: A callable that applies a non-linearity to its input.
        - norm1: A normalization module applied to the input.
        - conv1: A convolution module applied after the first normalization and non-linearity.
        - norm2: A normalization module applied after the first convolution.
        - conv2: A convolution module applied after the second normalization and non-linearity.
        - shortcut: An optional convolution module applied to the input for the residual connection.
        - resample: A flag indicating whether to resample the input. Resampling refers to the
            process of changing the spatial dimensions (height and width) of the input tensor. In
            this case, this would be downsampling through Conv mean.
    """

    non_linearity: Callable
    norm1: Union[eqx.nn.Identity, InstanceNorm2dPlus]
    conv1: eqx.nn.Conv2d
    norm2: Union[eqx.nn.Identity, InstanceNorm2dPlus]
    conv2: Union[eqx.nn.Conv2d, _ConvMeanPool]
    resample: bool
    shortcut: Optional[eqx.Module] = None

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        key: PRNGKeyArray,
        resample: bool = False,
        normalization: Union[
            Type[eqx.nn.Identity], Type[InstanceNorm2dPlus]
        ] = InstanceNorm2dPlus,
        act: Callable = jax.nn.elu,
        adjust_padding: bool = False,
        dilation: Optional[int] = None,
    ):
        """
        Initialize the _NCSNResidualBlock module with the given parameters.

        Args:
            - input_channels: The number of input channels.
            - output_channels: The number of output channels.
            - key: The key for the random number generator.
            - resample: If not None, resamples the input. Defaults to None.
            - normalization: The normalization function to use. Defaults to InstanceNorm2dPlus.
            - act: The activation function to use. Defaults to jax.nn.elu.
            - adjust_padding: If True, adjusts the padding of the input. Defaults to False.
            - dilation: If not None, applies dilated convolution. Defaults to None.

        Returns:
            - None
        """
        self.non_linearity = act
        self.resample = resample

        subkeys = jax.random.split(key, 5)

        if resample:
            if dilation is not None:
                self.conv1 = _dilated_conv3x3(
                    input_channels, input_channels, subkeys[0], dilation=dilation
                )
                self.norm2 = normalization(input_channels, subkeys[1])
                self.conv2 = _dilated_conv3x3(
                    input_channels, output_channels, subkeys[2], dilation=dilation
                )
                conv_shortcut: Callable[[int, int], eqx.Module] = partial(
                    _dilated_conv3x3, dilation=dilation, key=subkeys[3]
                )
            else:
                self.conv1 = _conv3x3(input_channels, input_channels, subkeys[0])
                self.norm2 = normalization(input_channels, subkeys[1])
                self.conv2 = _ConvMeanPool(
                    input_channels,
                    output_channels,
                    subkeys[2],
                    3,
                    adjust_padding=adjust_padding,
                )
                conv_shortcut = partial(
                    _ConvMeanPool,
                    kernel_size=1,
                    adjust_padding=adjust_padding,
                    key=subkeys[3],
                )
        else:
            if dilation is not None:
                conv_shortcut = partial(
                    _dilated_conv3x3, dilation=dilation, key=subkeys[0]
                )
                self.conv1 = _dilated_conv3x3(
                    input_channels, output_channels, subkeys[1], dilation=dilation
                )
                self.norm2 = normalization(output_channels, subkeys[2])
                self.conv2 = _dilated_conv3x3(
                    output_channels, output_channels, subkeys[3], dilation=dilation
                )
            else:
                conv_shortcut = partial(_convnxn_no_pad, key=subkeys[0])
                self.conv1 = _conv3x3(input_channels, output_channels, key=subkeys[1])
                self.norm2 = normalization(output_channels, subkeys[2])
                self.conv2 = _conv3x3(output_channels, output_channels, key=subkeys[3])

        if output_channels != input_channels or resample:
            self.shortcut = conv_shortcut(input_channels, output_channels)

        self.norm1 = normalization(input_channels, subkeys[4])

    def __call__(
        self, x: Float[Array, "input_channels h w"]
    ) -> Float[Array, "output_channels h w"]:
        """
        Perform the forward pass of the _NCSNResidualBlock module.

        Args:
            - x: The input tensor.

        Returns:
            - output: The output tensor after the sequence of operations
                and the residual connection.
        """
        if not (callable(self.norm1) and callable(self.norm2)):
            raise ValueError("norm1, norm2 must be callable")
        output = self.norm1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.norm2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.shortcut is not None:
            if not callable(self.shortcut):
                raise ValueError("self.shortcut must be callable")
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        return shortcut + output


class _RCUBlock(eqx.Module):
    """
    A module that implements a Residual Convolutional Unit (RCU) block.

    See figure 3 (b) of https://arxiv.org/abs/1611.06612.

    This class _RCUBlock is used to apply a sequence of operations including
    activation and convolution on the input tensor multiple times. The
    output is then added to the original input, forming a residual
    connection. This process is repeated for a number of blocks.

    Adapted from: https://github.com/ermongroup/ncsnv2/blob/master/models/layers.py#L112

    Attributes:
        - blocks: A list of lists of convolution modules. Each inner list represents a
            stage of operations in a block.
        - n_blocks: The number of blocks in the RCU.
        - n_stages: The number of stages in each block.
        - act: A callable that applies a non-linearity to its input.
    """

    blocks: List[List[eqx.nn.Conv2d]]
    n_blocks: int
    n_stages: int
    act: Callable

    def __init__(
        self,
        features: int,
        n_blocks: int,
        n_stages: int,
        key: PRNGKeyArray,
        act: Callable = jax.nn.relu,
    ) -> None:
        """
        Initialize the _RCUBlock module with the given parameters.

        Args:
            - features: The number of input and output channels.
            - n_blocks: The number of blocks in the RCU.
            - n_stages: The number of stages in each block.
            - key: The key for the random number generator.
            - act: The activation function to use. Defaults to jax.nn.relu.

        Returns:
            - None
        """
        blocks: List[List[eqx.nn.Conv2d]] = []
        for i in range(n_blocks):
            blocks.append([])
            for j in range(n_stages):
                blocks[i].append(
                    _conv3x3(features, features, stride=1, key=key, bias=False)
                )

        self.blocks = blocks
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        """
        Perform the forward pass of the _RCUBlock module.

        Args:
            - x: The input tensor.

        Returns:
            - output: The output tensor after the sequence of
                operations and the residual connections.
        """
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = self.blocks[i][j](x)
            x += residual
        return x


class _MSFBlock(eqx.Module):
    """
    A module that implements a Multi-Scale Fusion (MSF) block.

    See Fig 3 (c) of https://arxiv.org/abs/1611.06612.

    This class _MSFBlock is used to apply a 3x3 convolution on each input tensor, resize the
    convolved output to a target shape, and then sum up all the resized outputs.

    Adapted from: https://github.com/ermongroup/ncsnv2/blob/master/models/layers.py#L165

    Attributes:
        - convs: A list of convolution modules. Each convolution is
            applied to a corresponding input tensor.
        - features: The number of output channels for each convolution.
    """

    convs: List[eqx.nn.Conv2d]
    features: int

    def __init__(
        self, in_planes: Union[Tuple[int], List[int]], features: int, key: PRNGKeyArray
    ) -> None:
        """
        Initialize the _MSFBlock module with the given parameters.

        Args:
            - in_planes: A list of the number of input channels for each convolution.
            - features: The number of output channels for each convolution.
            - key: The key for the random number generator.

        Returns:
            - None
        """
        convs = []
        subkeys = jax.random.split(key, len(in_planes))
        for i, plane in enumerate(in_planes):
            convs.append(_conv3x3(plane, features, stride=1, bias=True, key=subkeys[i]))
        self.convs = convs
        self.features = features

    def __call__(
        self, xs: List, shape: Tuple[int, int]
    ) -> Float[Array, "features shape[0] shape[1]"]:
        """
        Perform the forward pass of the _MSFBlock module.

        Args:
            - xs: A list of input tensors (Float[Array, "in_planes h w"]).
            - shape: The target shape to resize the convolved outputs to.

        Returns:
            - output: The output tensor after the convolutions, resizing, and summation.
        """
        sums = jnp.zeros((self.features, *shape))
        for i, x in enumerate(xs):
            h = self.convs[i](x)
            h = jax.image.resize(h, (1, *shape), method="bilinear")
            sums += h
        return sums


class _CRPBlock(eqx.Module):
    """
    A module that implements a Chained Residual Pooling (CRP) block.

    See Fig 3 (d) of https://arxiv.org/abs/1611.06612.

    This class _CRPBlock is used to apply a sequence of operations including activation,
    max/avg pooling, and convolution on the input tensor. The output is then added to
    the original input, forming a residual connection. This process is repeated
    for a number of stages.

    Adapted from: <source code link>

    Attributes:
        - act: A callable that applies a non-linearity to its input.
        - maxpool: A max/avg pooling module.
        - convs: A list of convolution modules. Each convolution is
            applied after a pooling operation.
    """

    act: Callable
    maxpool: Union[eqx.nn.MaxPool2d, eqx.nn.AvgPool2d]
    convs: List[eqx.nn.Conv2d]

    def __init__(
        self,
        features: int,
        n_stages: int,
        key: PRNGKeyArray,
        act: Callable = jax.nn.relu,
        maxpool: bool = True,
    ) -> None:
        """
        Initialize the _CRPBlock module with the given parameters.

        Args:
            - features: The number of input and output channels for each convolution.
            - n_stages: The number of stages in the CRP block.
            - key: The key for the random number generator.
            - act: The activation function to use. Defaults to jax.nn.relu.
            - maxpool: If True, uses max pooling; otherwise, uses
                average pooling. Defaults to True.

        Returns:
            - None
        """
        convs = []
        subkeys = jax.random.split(key, n_stages)
        for i in range(n_stages):
            convs.append(
                _conv3x3(features, features, stride=1, bias=False, key=subkeys[0])
            )
        self.convs = convs
        if maxpool:
            self.maxpool = eqx.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.maxpool = eqx.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self.act = act

    def __call__(self, x: Float[Array, "features h w"]) -> Float[Array, "features h w"]:
        """
        Perform the forward pass of the _CRPBlock module.

        Args:
            - x: The input tensor.

        Returns:
            - output: The output tensor after the sequence of
                operations and the residual connections.
        """
        x = self.act(x)
        path = x
        for conv in self.convs:
            path = self.maxpool(path)
            path = conv(path)
            x = path + x
        return x


class _NCSNRefineBlock(eqx.Module):
    """
    A module that implements a Refine block used in the NCSNv2 model.

    This class _NCSNRefineBlock is used to apply a sequence of operations including
    RCU blocks, MSF block (if not the start), CRP block, and another RCU block
    on the input tensors.

    Adapted from: https://github.com/ermongroup/ncsnv2/blob/master/models/layers.py#L214

    Attributes:
        - n_blocks: The number of input tensors.
        - adapt_convs: A list of _RCUBlock modules. Each module is applied to a
            corresponding input tensor.
        - output_convs: A _RCUBlock module applied at the end.
        - crp: A _CRPBlock module applied after the MSF block.
        - msf: An optional _MSFBlock module applied if not the start of the
            refine blocks.
    """

    n_blocks: int
    adapt_convs: List[_RCUBlock]
    output_convs: _RCUBlock
    crp: _CRPBlock
    msf: Optional[_MSFBlock] = None

    def __init__(
        self,
        in_planes: Union[Tuple[int], List[int]],
        features: int,
        key: PRNGKeyArray,
        act: Callable = jax.nn.relu,
        start: bool = False,
        end: bool = False,
        maxpool: bool = True,
    ):
        """
        Initialize the _NCSNRefineBlock module with the given parameters.

        Args:
            - in_planes: A list of the number of input channels for each RCU block.
            - features: The number of output channels for the RCU and CRP blocks.
            - key: The key for the random number generator.
            - act: The activation function to use. Defaults to jax.nn.relu.
            - start: If True, does not apply the MSF block. Defaults to False.
            - end: If True, applies 3 RCU blocks at the end; otherwise,
                applies 1. Defaults to False.
            - maxpool: If True, uses max pooling in the CRP block; otherwise,
                uses average pooling. Defaults to True.

        Returns:
            - None
        """
        if not isinstance(in_planes, tuple) and not isinstance(in_planes, list):
            raise TypeError("in_planes must be a tuple or list!")

        adapt_convs = []
        n_blocks = len(in_planes)
        self.n_blocks = n_blocks
        subkeys = jax.random.split(key, n_blocks + 1)
        key = subkeys[-1]
        for i in range(n_blocks):
            adapt_convs.append(_RCUBlock(in_planes[i], 2, 2, subkeys[i], act))
        self.adapt_convs = adapt_convs
        key, subkey = jax.random.split(key)
        self.output_convs = _RCUBlock(features, 3 if end else 1, 2, subkey)

        if not start:
            key, subkey = jax.random.split(key)
            self.msf = _MSFBlock(in_planes, features, subkey)

        key, subkey = jax.random.split(key)
        self.crp = _CRPBlock(features, 2, subkey, act, maxpool)

    def __call__(
        self, xs: List, ouput_shape: Tuple[int, int]
    ) -> Float[Array, "features output_shape[0] output_shape[1]"]:
        """
        Perform the forward pass of the _NCSNRefineBlock module.

        Args:
            - xs: A list of input tensors ([Float[Array, ["c h w"]]]).
            - ouput_shape: The target shape to resize the outputs to.

        Returns:
            - output: The output tensor after the sequence of operations.
        """
        hs = []
        for i, x in enumerate(xs):
            h = self.adapt_convs[i](x)
            hs.append(h)

        if self.n_blocks > 1:
            if self.msf is None:
                raise TypeError("n_blocks > 1 but start is True!")
            h = self.msf(hs, ouput_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h


class NCSNv2(eqx.Module):
    """
    The full NCSNv2 architecture.

    This class NCSNv2 is used to apply a sequence of operations including convolution,
    normalization, activation, and residual blocks on the input tensor. The output
    is then refined through a series of refine blocks.

    Adapted from: https://github.com/ermongroup/ncsnv2/blob/master/models/ncsnv2.py#L11

    Attributes:
        - res1, res2, res3, res4: Lists of _NCSNResidualBlock modules. Each list
            represents a layer of residual blocks.
        - refine1, refine2, refine3, refine4: _NCSNRefineBlock modules used to
            refine the output of the residual blocks.
        - logit_transform: A flag indicating whether to apply logit
            transformation on the input.
        - rescaled: A flag indicating whether the input has been rescaled.
        - act: A callable that applies a non-linearity to its input.
        - begin_conv: A convolution module applied at the beginning.
        - end_conv: A convolution module applied at the end.
        - norm: A normalization module applied before the final
            activation and convolution.
    """

    res1: List[_NCSNResidualBlock]
    res2: List[_NCSNResidualBlock]
    res3: List[_NCSNResidualBlock]
    res4: List[_NCSNResidualBlock]
    refine1: _NCSNRefineBlock
    refine2: _NCSNRefineBlock
    refine3: _NCSNRefineBlock
    refine4: _NCSNRefineBlock
    rescaled: bool
    act: Callable
    begin_conv: eqx.nn.Conv2d
    end_conv: eqx.nn.Conv2d
    norm: eqx.Module

    def __init__(self, config: Dict[str, Any], key: PRNGKeyArray) -> None:
        """
        Initialize the NCSNv2 module with the given parameters.

        Args:
            - config: A dictionary containing the configuration parameters.
            - key: The key for the random number generator.

        Returns:
            - None
        """
        self.rescaled = config["rescaled"]
        self.act = config["act"]
        ngf = config["ngf"]
        act = config["act"]
        subkeys = jax.random.split(key, 15)
        self.begin_conv = eqx.nn.Conv2d(
            config["channels"], ngf, 3, stride=1, padding=1, key=subkeys[0]
        )
        self.norm = InstanceNorm2dPlus(ngf, subkeys[1])
        self.end_conv = eqx.nn.Conv2d(
            ngf, config["channels"], 3, stride=1, padding=1, key=subkeys[2]
        )
        self.res1 = [
            _NCSNResidualBlock(ngf, ngf, subkeys[3], resample=False, act=act),
            _NCSNResidualBlock(ngf, ngf, subkeys[4], resample=False, act=act),
        ]
        self.res2 = [
            _NCSNResidualBlock(ngf, ngf * 2, subkeys[5], resample=True, act=act),
            _NCSNResidualBlock(ngf * 2, ngf * 2, subkeys[6], resample=False, act=act),
        ]
        self.res3 = [
            _NCSNResidualBlock(
                ngf * 2, ngf * 2, subkeys[7], resample=True, act=act, dilation=2
            ),
            _NCSNResidualBlock(
                ngf * 2, ngf * 2, subkeys[8], resample=False, act=act, dilation=2
            ),
        ]
        if config["image_size"] == 28:
            self.res4 = [
                _NCSNResidualBlock(
                    2 * ngf,
                    2 * ngf,
                    subkeys[9],
                    resample=True,
                    act=act,
                    adjust_padding=True,
                    dilation=4,
                ),
                _NCSNResidualBlock(
                    2 * ngf, 2 * ngf, subkeys[10], resample=False, act=act, dilation=4
                ),
            ]
        else:
            self.res4 = [
                _NCSNResidualBlock(
                    2 * ngf,
                    2 * ngf,
                    subkeys[9],
                    resample=True,
                    act=act,
                    adjust_padding=False,
                    dilation=4,
                ),
                _NCSNResidualBlock(
                    2 * ngf, 2 * ngf, subkeys[10], resample=False, act=act, dilation=4
                ),
            ]
        self.refine1 = _NCSNRefineBlock(
            [ngf * 2], ngf * 2, key=subkeys[11], act=act, start=True
        )
        self.refine2 = _NCSNRefineBlock(
            [ngf * 2, ngf * 2], ngf * 2, key=subkeys[12], act=act
        )
        self.refine3 = _NCSNRefineBlock(
            [ngf * 2, ngf * 2], ngf, key=subkeys[13], act=act
        )
        self.refine4 = _NCSNRefineBlock(
            [ngf, ngf], ngf, key=subkeys[14], act=act, end=True
        )

    def _compute_cond_module(
        self, module: List[_NCSNResidualBlock], x: Float[Array, "c h w"]
    ) -> Float[Array, "new_channels h w"]:
        """
        Apply a list of residual blocks on the input tensor.

        Args:
            - module: A list of residual blocks.
            - x: The input tensor.

        Returns:
            - output: The output tensor after the residual blocks.
        """
        for m in module:
            if not callable(m):
                raise ValueError("module component was not callable")
            x = m(x)
        return x

    def __call__(self, x: Float[Array, "channels h w"]) -> Float[Array, "channels h w"]:
        """
        Perform the forward pass of the NCSNv2 module.

        Args:
            - x: The input tensor.
        Returns:
            - output: The output tensor after the sequence of operations
                and the refine blocks.
        """
        if not self.rescaled:
            h = 2 * x - 1.0
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[1:])  # type: ignore[arg-type]
        ref2 = self.refine2([layer3, ref1], layer3.shape[1:])  # type: ignore[arg-type]
        ref3 = self.refine3([layer2, ref2], layer2.shape[1:])  # type: ignore[arg-type]
        output = self.refine4([layer1, ref3], layer1.shape[1:])  # type: ignore[arg-type]

        if not callable(self.norm):
            raise ValueError("self.norm is not callable")
        output = self.norm(output)
        output = self.act(output)
        output = self.end_conv(output)

        return output
