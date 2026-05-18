from warnings import warn

from net.models import cuda_lung_conv_network
from net.models import mnist_conv_network
from net.models import multi_layer_perceptron
from net.models import myeloma_mlp


def main():

    try:
        myeloma_mlp.main()
    except FileNotFoundError as e:
        warn(f"{'\n' * 2}{'=' * 30}{'\n'}{e}{'\n'}{'=' * 30}{'\n' * 2}")

    try:
        cuda_lung_conv_network.main()
    except FileNotFoundError as e:
        warn(f"{'\n' * 2}{'=' * 30}{'\n'}{e}{'\n'}{'=' * 30}{'\n' * 2}")

    multi_layer_perceptron.main()
    mnist_conv_network.main()


if __name__ == '__main__':
    main()
