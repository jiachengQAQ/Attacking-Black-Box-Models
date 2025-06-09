import os
import argparse
import wandb
import torch
import torch.optim as optim
from helper.utils import setup_seed, train, test, convert_tensor_to_image
from helper.dataset import get_dataset, cifar10_dataloader, cifar100_dataloader
#from helper.models import get_model
from custom_dataloader import CustomDataset, ImageNettee_CustomDataset, Test_Dataset
import pdb
from nets import resnet34, CNN, CNNCifar10, resnet18, resnet50, MLP


def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training with scratch')
    # Training model hyperparameter settings
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=120, help="number of training epochs")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
    parser.add_argument('--weight_decay', '--wd', default=5e-4,
                        type=float, metavar='W')
    parser.add_argument("--lr_scheduler", default="cosine", choices=["cosine", "step"])
    parser.add_argument('--num_workers', type=int, default=16, help="number of workers for data loading")
    # model setting
    parser.add_argument('--net', type=str,
                        choices=['resnet18', 'resnet34'],
                        default="resnet18",
                        help='model name to train')
    parser.add_argument('--net_path', type=str,
                        default = None,
                        help='load model weight path')
    # dataset setting
    parser.add_argument('--data_type', type=str,
                        choices=["cifar10", "cifar100", "domainnet", "imagenet100", "imagenette", "imagefruit", "imageyellow", "imagesquawk"],
                        default="imagenette",
                        help='data set type')
    parser.add_argument('--data_path_train', default=None, type=str, help='data path for train')
    parser.add_argument('--data_path_test', default=None, type=str, help='data path for test')
    parser.add_argument('--sample_data_nums', default=None, type=int, help='sample number of syn images if None samples all data')
    parser.add_argument('--syn', action='store_true', default=False,
                        help='if syn dataset')
    parser.add_argument('--train_syn_data', default=False,
                        help='if train on custm syn_data')
    parser.add_argument('--train_imb_data', default=False,
                        help='if train on imblance data')
    parser.add_argument('--if_blip', action='store_true', default=False,
                        help='if blip syn dataset')
    # domainnet dataset setting
    parser.add_argument('--labels', nargs='+', type=int,
                        default=[953, 954, 949, 950, 951, 957, 952, 945, 943, 948], #['airplane', 'clock', 'axe', 'basketball', 'bicycle', 'bird', 'strawberry', 'flower', 'pizza', 'bracelet'],
                        help='domainnet subdataset labels')
    parser.add_argument('--domains', nargs='+', type=str,
                        default=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
                        help='domainent domain')
    # others setting
    parser.add_argument('--seed', type=int, default=0, help="random seed for reproducibility")
    parser.add_argument('--exp_name', type=str, default="exp_1",
                        help="the name of this run")
    parser.add_argument('--wandb', type=int, default=1,
                        help="set 1 for wandb logging")
    parser.add_argument('--device', type=str, default="cuda:0",
                        choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
                        help="start number of generated images")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get args
    args = get_arguments()

    # mkdir for save models
    save_tmp = "{}-{}-{}".format(args.net, args.data_type, args.train_syn_data)
    if args.train_syn_data:
        save_tmp = save_tmp + "-syn"
    elif args.train_imb_data:
        save_tmp = save_tmp + "-imb"
    save_model_dir = os.path.join("weights_non_normalize", save_tmp)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # setting
    setup_seed(args.seed)

    # wandb
    if args.wandb == 1:
        wandb.init(config=args, project="train from scratch", group="train_from_scratch", name=save_tmp)

    # getdataset
    if args.data_type == "cifar10":
        train_set, test_set = cifar10_dataloader(data_dir="")
        if args.train_syn_data:
            root_dir = ""
            custom_train_dataset = CustomDataset(root_dir, args)
            trainloader = torch.utils.data.DataLoader(custom_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=16)

        else:
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers)
    elif args.data_type == "cifar100":
        train_set, test_set = cifar100_dataloader(data_dir="cifar-100-python")
        if args.train_syn_data:
            root_dir = ""
            custom_train_dataset = CustomDataset(root_dir)
            trainloader = torch.utils.data.DataLoader(custom_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=16)

        else:
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers)


    else:
        if args.data_type == "imagefruit":
            args.labels = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]
        elif args.data_type == "imagesquawk":
            args.labels = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]
        elif args.data_type == "imageyellow":
            args.labels = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]
        elif args.data_type == "imagenet100":
            args.labels = [117, 70, 88, 133, 5, 97, 42, 60, 14, 3, 130, 57, 26, 0, 89, 127, 36, 67, 110, 65, 123, 55,
                           22, 21, 1, 71,
                           99, 16, 19, 108, 18, 35, 124, 90, 74, 129, 125, 2, 64, 92, 138, 48, 54, 39, 56, 96, 84, 73,
                           77, 52, 20,
                           118, 111, 59, 106, 75, 143, 80, 140, 11, 113, 4, 28, 50, 38, 104, 24, 107, 100, 81, 94, 41,
                           68, 8, 66,
                           146, 29, 32, 137, 33, 141, 134, 78, 150, 76, 61, 112, 83, 144, 91, 135, 116, 72, 34, 6, 119,
                           46, 115, 93, 7]
        elif args.data_type == "imagenette":
                        args.labels = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
                                
        trainset = get_dataset(data_type=args.data_type,
                                if_syn=args.syn,
                                if_train=True,
                                data_path=args.data_path_train,
                                sample_data_nums=args.sample_data_nums,
                                seed=args.seed,
                                if_blip=args.if_blip,
                                labels=args.labels,
                                domains=args.domains)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        testset = get_dataset(data_type=args.data_type,
                                if_syn=False,
                                if_train=False,
                                data_path=args.data_path_test,
                                sample_data_nums=None,
                                seed=args.seed,
                                if_blip=False,
                                labels=args.labels,
                                domains=args.domains)

        if args.data_type == "domainnet":
            testloader = {}
            # testset is a dict
            for domain in testset:
                testloader[domain] = torch.utils.data.DataLoader(testset[domain], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        else:
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # get model
    # if "imagenet100" in args.data_type:
    #     net = get_model(args.net, args.net_path, num_classes=100)
    if args.net == "resnet18":
        if args.data_type == "cifar10":
            net = resnet18(num_classes=10)
        else:
            net = resnet18(num_classes=100)
    else:
        if args.data_type == "cifar10" or args.data_type == "imagenette" or args.data_type == "imagefruit" or args.data_type == "imageyellow" or args.data_type == "imagesquawk":
            net = resnet34(num_classes=10)
        else:
            net = resnet34(num_classes=100)
    net.to(args.device)
    #net = torch.nn.DataParallel(net).to('cuda')

    # train
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=args.momentum, weight_decay=args.weight_decay)
    # Learning Rate Scheduler
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[int(args.n_epochs * 0.5), int(args.n_epochs * 0.75)],
                                                         gamma=0.1)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    else:
        raise ValueError
    best_acc = 0
    best_acc_domain = {}
    for epoch in range(args.n_epochs):

        print("train epoch {}/{}".format(epoch, args.n_epochs))
        correct, total, train_loss = train(net, trainloader, optimizer, args, if_log=True)
        wandb.log({"epoch": epoch, "train_loss": train_loss / args.batch_size})
        wandb.log({"epoch": epoch, "train_acc": 100.*correct/total})
        scheduler.step()

        print("test epoch {}/{}".format(epoch, args.n_epochs))
        if args.data_type == "domainnet":
            correct_all, total_all = 0, 0
            acc_domain = dict()
            for domain in args.domains:
                correct, total, test_loss = test(net, testloader[domain])
                correct_all += correct
                total_all += total
                acc_domain[domain] = round(100.0 * correct / total, 2)
                if args.wandb == 1:
                    wandb.log({"epoch": epoch, "test_{}_loss".format(domain): test_loss/args.batch_size})
                    wandb.log({"epoch": epoch, "test_{}_acc".format(domain): acc_domain[domain]})

            acc = 100.0 * correct_all / total_all
            if args.wandb == 1:
                wandb.log({"test_acc":acc})

            if acc > best_acc:
                best_acc = acc
                best_acc_domain = acc_domain
                torch.save(net.state_dict(),
                    os.path.join(save_model_dir, 'model-best-epoch-best.pt'))

            print("epoch/max_epoch:{}/{}  test_loss:{}  acc_domain/best_acc_domain:{}/{} \n test_acc/best_acc:{}/{}"
                    .format(epoch, args.n_epochs, round(test_loss, 2), acc_domain, best_acc_domain, round(acc, 2), round(best_acc,2 )))
        else:
            correct, total, test_loss = test(net, testloader, args,if_log=True)
            acc = 100.0 * correct / total
            if args.wandb == 1:
                wandb.log({"epoch": epoch,"test_loss": test_loss/args.batch_size})
                wandb.log({"epoch": epoch,"test_acc":acc})

            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(),
                    os.path.join(save_model_dir, 'model-best-epoch-best.pt'))
                print("save new model here")
            print("epoch/max_epoch:{}/{} test_loss:{} \n test_acc/best_acc:{}/{}"
                    .format(epoch, args.n_epochs, round(test_loss, 2), round(acc, 2), round(best_acc, 2)))
