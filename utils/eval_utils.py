
import torch
def evaluate(net, dloader_dt):
    net.eval()
    acc = 0
    nums = 0
    # dloader_dt = dloaders[0]  # Dt
    with torch.no_grad():
        for (x, y, _) in dloader_dt:
            x = x.cuda().float()
            y = torch.tensor(list(y)).cuda().long()
            z = net.predict(x)
            pred = z.data.max(1)[1]
            acc += pred.eq(y).sum()
            nums += y.shape[0]

    acc = int(acc) / nums
    print("[Accuracy = {:.2%}]".format(acc))
    return acc

def eval_dt_multi_classifier(eval_net_list, test_dloader, baseline='other'):
    acc = 0
    nums = 0  #
    with torch.no_grad():
        for net in eval_net_list:
            net.eval()
        if baseline != 'ViT_Cross' or baseline != 'ViT_Cross_random_offset':
            for (x, y, _) in test_dloader:
                x = x.cuda().float()
                y = torch.tensor(list(y)).cuda().long()

                z = [net.predict(x).unsqueeze(1) for net in eval_net_list]
                z_sum = torch.cat((z), 1).sum(1)  # [bs, n_net, n_classes] -> [bs, n_classes]
                pred = z_sum.data.max(1)[1]
                acc += pred.eq(y).sum()
                nums += y.shape[0]
        else:
            for (x, y) in test_dloader:
                x = x.cuda().float()
                y = torch.tensor(list(y)).cuda().long()

                z = [net.predict(x).unsqueeze(1) for net in eval_net_list]
                z_sum = torch.cat((z), 1).sum(1)  # [bs, n_net, n_classes] -> [bs, n_classes]
                pred = z_sum.data.max(1)[1]
                acc += pred.eq(y).sum()
                nums += y.shape[0]
    acc = int(acc) / nums
    print("[Accuracy = {:.2%}]".format(acc))
    return acc
