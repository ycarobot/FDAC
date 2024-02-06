
import numpy as np
import torch


def fa_Cross_KL(net_weight_da_list, ds_net_list):
    # Copy the parameters to Ds after "DA train with dt"  
    # n_ds classifiers
    for i in range(len(ds_net_list)):
        ds_net_list[i].load_state_dict(net_weight_da_list[i])
    return ds_net_list


def fa_1Dt_multiDs(net_list, domain_weight=None):
    if domain_weight is None:
        domain_weight = [(1 / len(net_list)) for _ in range(len(net_list))]
    # do not update Classifier
    named_parameter_list = [net.named_parameters() for net in net_list]  # (k, v)
    for parameter_list in zip(*named_parameter_list):  #Return the name and parameters
        # Get the parameters of all the models that are in the same level
        # Sort by default 
        source_parameters = [
            domain_weight[idx] * parameter[1].data.clone() for idx, parameter in enumerate(parameter_list)]
        # 0 for Dt.  1 for Ds. The second index represents the value
        parameter_list[0][1].data = sum(source_parameters)
        for parameter in parameter_list[1:]: # Copy from Dt to Ds
            # If the level is updated by index, try like data[4:6] 
            parameter[1].data = parameter_list[0][1].data.clone()
    return net_list


def ds_to_dt(net_list, domain_weight=None):
    if domain_weight is None:
        domain_weight = [(1 / len(net_list)) for _ in range(len(net_list))]
    # do not update Classifier
    named_parameter_list = [net.named_parameters() for net in net_list]  # (k, v)
    for parameter_list in zip(*named_parameter_list):  # Return name and parameters
        # Get the parameters of all the models that are in the same level
        # Sort by default 
        source_parameters = [
            domain_weight[idx] * parameter[1].data.clone() for idx, parameter in enumerate(parameter_list) if idx != 0]
        # 0 for Dt.  1 for Ds. The second index represents the value
        parameter_list[0][1].data = sum(source_parameters)
        for parameter in parameter_list[1:]: #  Copy from Dt to Ds
            # If the level is updated by index, try like data[4:6] 
            parameter[1].data = parameter_list[0][1].data.clone()
    return net_list


def vit_block_name(net, block_index_list):
    k_list = ['backbone.blocks.{}.'.format(int(i)) for i in block_index_list]
    name_list = []
    for k, v in net.named_parameters():
        if k[:len(k_list[0])] in k_list:
            name_list.append(k)
    return name_list

# update params
def federated_update_parameters_Blocks(net_list, block_index_list=None, domain_weight=None):
    if domain_weight is None:
        domain_weight = [(1 / len(net_list)) for _ in range(len(net_list))]
        # # for source only
        # domain_weight = [(1 / len(net_list)) for _ in range(len(net_list[1:]))]
        # domain_weight.insert(0, 0.0)
    if block_index_list is None:
        block_index_list = np.arange(12)
    blocks_k_list = vit_block_name(net_list[0], block_index_list)
    for k, v in net_list[0].named_parameters():
        if 'backbone.blocks' in k and k not in blocks_k_list:
                continue  # not update these block
        else:
            temp = torch.zeros_like(v)
            for i in range(len(net_list)):
                temp += domain_weight[i] * net_list[i].state_dict()[k]
            net_list[0].state_dict()[k].data.copy_(temp) # dt
            for i in range(1, len(net_list)):
                net_list[i].state_dict()[k].data.copy_(net_list[0].state_dict()[k])
    return net_list


def federated_average(net_list):
    domain_weight = [1 / len(net_list[1:]) for _ in range(len(net_list[1:]))] # do not consider dt
    for k, v in net_list[0].named_parameters():
        temp = torch.zeros_like(v)
        for i in range(1, len(net_list)):
            temp += domain_weight[i-1] * net_list[i].state_dict()[k]
            net_list[0].state_dict()[k].data.copy_(temp)
    return net_list






if __name__ == '__main__':
    from baseline.ViT_network.network import ViT_diversity
    net = ViT_diversity(n_classes=30)
    # for k, v in net.named_parameters():
    #     print(k)
    name_list = vit_block_name(net, [1, 2, 3])
    federated_update_parameters_Blocks([net, net], block_index_list=[0,1,2,3,4,5,6,7])
