import os
import sys

from net.model.utility.my_torchsummary import summary_string


def debug_network_summary(net, backbone, input_size, batch_size, device):
    """ | --------------------- |
        | DEBUG NETWORK SUMMARY |
        | --------------------- |

        show and save
        - summary network

    """

    print("\nNETWORK SUMMARY"
          "\nBackbone: {}"
          "\nClassification Model"
          "\nRegression Model".format(backbone))

    network_summary_filename = "GravityNet-microcalcifications-models-summary|backbone={}.txt".format(backbone)
    network_summary_path = os.path.join("./debug", network_summary_filename)

    # torch.summary network
    # summary(net, input_size)

    # save torch.summary network
    with open(network_summary_path, 'w') as f:
        report, _ = summary_string(model=net,
                                   input_size=input_size,
                                   batch_size=batch_size,
                                   device=device)
        f.write("GravityNet-microcalcifications | SUMMARY NETWORK\n")
        f.write("Backbone: {}\n".format(backbone))
        f.write("Input Size: {}\n".format(input_size))
        f.write("Batch Size: {}\n".format(batch_size))
        f.write("\n")
        f.write(report)

    str_err = "\nDEBUG NETWORK: COMPLETE"
    sys.exit(str_err)
