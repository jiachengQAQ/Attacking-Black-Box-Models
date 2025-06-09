import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from custom_dataloader import CustomDataset, Imagenet_Dataset
from torchvision.transforms import transforms
import torchvision
import torch.nn as nn
import pdb
import argparse
import wandb
from nets import resnet34, CNN, CNNCifar10, resnet18, resnet50, MLP
from helper.dataset import get_dataset, cifar10_dataloader
from advertorch.attacks import GradientSignAttack, PGDAttack, LinfPGDAttack
from advertorch.attacks import LinfPGDAttack, LinfBasicIterativeAttack
import os
from collections import OrderedDict
from copy import deepcopy
from utils import *


def load_data(args):
    #setup_seed(1024)

    if args.data_type == "cifar10":
        root_dir = "./syn_cifar10_"+args.query_number
        custom_train_dataset = CustomDataset(root_dir,args)
        trainloader = torch.utils.data.DataLoader(custom_train_dataset, batch_size=128, shuffle=True,
                                                  num_workers=16)
        test_transform = transforms.Compose([
            transforms.ToTensor(),

        ])
        testdataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=128, shuffle=True,
                                                  num_workers=16)
    elif args.data_type == "cifar100":
        root_dir = "./syn_cifa100_"+args.query_number
        custom_train_dataset = CustomDataset(root_dir,args)
        trainloader = torch.utils.data.DataLoader(custom_train_dataset, batch_size=128, shuffle=True,
                                                  num_workers=16)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        testdataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=128, shuffle=True,
                                                  num_workers=16)
    else:
        root_dir = "./syn_imagenette_"+args.query_number
        test_root_dir = "Imagenet/val_classify/val_classify"
        if args.data_type == "imagefruit":
            args.labels = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]
        elif args.data_type == "imagesquawk":
            args.labels = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]
        elif args.data_type == "imageyellow":
            args.labels = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]
        elif args.data_type == "imagenette":
            args.labels = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

        custom_train_dataset = Imagenet_Dataset(root_dir, "train")
        real_test_dataset = get_dataset(data_type=args.data_type,
                                if_syn=False,
                                if_train=False,
                                data_path=test_root_dir,
                                sample_data_nums=None,
                                seed=args.seed,
                                if_blip=False,
                                labels=args.labels,
                                domains=args.domains)

        trainloader = torch.utils.data.DataLoader(custom_train_dataset, batch_size=128, shuffle=True,
                                                  num_workers=16)
        testloader = torch.utils.data.DataLoader(real_test_dataset, batch_size=128, shuffle=False,
                                                  num_workers=16)
    return trainloader, testloader
class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()
    def forward(self, teacehr_labels, teacher_logits, student_logits, args):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        """
        if args.loss == "l1":
            loss_fn = F.l1_loss
            loss = loss_fn(student_logits, teacher_logits.to('cuda'))
        elif args.loss == "ce":
            loss = F.cross_entropy(student_logits, teacehr_labels)

        return loss

# Training loop for knowledge distillation
def train_student_model(student_model, distillation_loss, train_loader, teacher_prob_dict, teacher_label_dict, teacher_logits_dict, num_epochs, optimizer, args):
    setup_seed(1024)
    train_loss = 0
    correct = 0
    total = 0
    student_model.train()

    for batch_idx, (inputs, targets1) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets1 = inputs.to('cuda'), targets1.to('cuda')
        # Forward pass on student model
        student_logits = student_model(inputs)
        student_scores = F.softmax(student_logits, dim=1)

        # get logits, prob, label from teacher
        teacher_logits = teacher_logits_dict[batch_idx]
        teacher_scores = teacher_prob_dict[batch_idx]
        teacher_labels = teacher_label_dict[batch_idx]

        loss = distillation_loss( teacher_labels, teacher_logits, student_logits,
                                 args)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += targets1.size(0)
        correct += predicted.eq(targets1).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Train Loss Student: %.3f | Acc: %.3f%% (%d/%d)'
                          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return correct, total, train_loss



def launch_attack(student_model, teacher_model,test_loader, target,args):
    setup_seed(1024)

    cfgs = dict(random=True, test_num_steps=40, test_step_size=0.01, test_epsilon=16/255, num_classes=10)

    if args.data_type == "cifar10" or args.data_type == "cifar100":
        cfgs = dict(test_step_size=2.0 / 255, test_epsilon=8.0 / 255)

    if args.attack_type == "BIM":
        adversary = LinfBasicIterativeAttack(
            student_model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=cfgs['test_epsilon'],
            nb_iter=120, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
            targeted=target)
    elif args.attack_type == 'PGD':
        if args.target:
            adversary = PGDAttack(
                student_model,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=cfgs['test_epsilon'],
                nb_iter=20, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
                targeted=target)
        else:
            adversary = PGDAttack(
                student_model,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=cfgs['test_epsilon'],
                nb_iter=20, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
                targeted=target)
    # FGSM
    elif args.attack_type == 'FGSM':

        adversary = GradientSignAttack(
            student_model,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=cfgs['test_epsilon'],
            targeted=target)



    student_model.eval()
    correct = 0.0
    uncorrect = 0.0
    correct_ghost = 0.0
    
    teacher_model.eval()
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        teacher_outputs_adv = teacher_model(inputs)
        _, predicted = teacher_outputs_adv.max(1)
        inputs.requires_grad = True
        if target:
            if args.data_type=="cifar10":
                labels = torch.randint(0, 9, (inputs.shape[0],)).cuda()
            else:
                labels = torch.randint(0, 99, (inputs.shape[0],)).cuda()
            idx = torch.where(predicted != labels)[0]
            uncorrect += idx.shape[0]
            adv_inputs_ghost = adversary.perturb(inputs[idx], labels[idx])
            with torch.no_grad():
                teacher_outputs_adv = teacher_model(adv_inputs_ghost)
                _, predicted_adv = teacher_outputs_adv.max(1)
                correct_ghost += (predicted_adv == labels[idx]).sum()
        else:
            idx = torch.where(predicted == labels)[0]
            correct += idx.shape[0]
            adv_inputs_ghost = adversary.perturb(inputs[idx], labels[idx])
            with torch.no_grad():
                teacher_outputs_adv = teacher_model(adv_inputs_ghost)
                _, predicted_adv = teacher_outputs_adv.max(1)
                correct_ghost += (predicted_adv != labels[idx]).sum()
    return 100. * correct_ghost / uncorrect if target else 100. * correct_ghost / correct

def test(student_model,test_loader, args):
    student_model.eval()
    criterion = nn.CrossEntropyLoss()
    correct_s = 0
    total = 0
    loss_s = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            # Forward pass on student model
            student_outputs = student_model(inputs)
            total += targets.size(0)
            _, predicted_s = student_outputs.max(1)
            loss_s = criterion(student_outputs, targets)
            loss_s += loss_s.item()
            correct_s += predicted_s.eq(targets).sum().item()
        acc = 100.0 * correct_s / total
        print('Test Accuarcy : {}'.format(acc))
        return acc, loss_s
    
def get_arguments():

    parser = argparse.ArgumentParser(description='PyTorch Training with pretrain weight')
    # Training model hyperparameter settings
    parser.add_argument('--data_type', type=str,
                        choices=["cifar10", "cifar100"],
                        default="cifar10",
                        help='data set type')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=120, help="number of training epochs")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size for training")
    parser.add_argument('--weight_decay', '--wd', default=5e-4,
                        type=float, metavar='W')
    parser.add_argument('--net', type=str,
                        choices=['resnet18', 'resnet34'],
                        default="resnet18",
                        help='model name to train')
    parser.add_argument('--wandb', type=int, default=1)
    parser.add_argument("--lr_scheduler", default="cosine", choices=["cosine", "step"])
    parser.add_argument("--loss", default="l1", choices=["l1", "ce"])
    parser.add_argument('--seed', type=int, default=1024, help="random seed for reproducibility")
    parser.add_argument('--load_parallel', default=False)
    parser.add_argument('--query_number', default='5000', type=str)
    parser.add_argument("--attack_type", default='BIM', type=str,
                        choices=['BIM', 'PGD', 'FGSM'])
    parser.add_argument('--target', default='Untarget', type=str, choices=['Untarget', 'Target'])
    parser.add_argument("--only_dfme", default=False)
    parser.add_argument("--save_result", default=False)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = get_arguments()
    setup_seed(args.seed)
    save_tmp = "Attack-{}-{}-{}-{}-{}".format(args.data_type, args.query_number, args.attack_type, args.target, args.loss)
    save_model_dir = os.path.join("weights_attack", save_tmp)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    # wandb
    if args.wandb == 1:
        wandb.init(config=args, project="train from scratch", group="train_from_scratch", name=save_tmp)

    if args.data_type == "cifar10":
        teacher_model = resnet34(num_classes=10)
        student_model = resnet18(num_classes=10)
    else:
        teacher_model = resnet34(num_classes=100)
        student_model = resnet18(num_classes=100)

    # Load teacher and student model
    student_model, teacher_model = load_model(teacher_model, student_model, args)
    student_model = torch.nn.DataParallel(student_model).to('cuda')
    teacher_model = torch.nn.DataParallel(teacher_model).to('cuda')
    
    # Load data
    train_loader, test_loader = load_data(args)

    
    # Only inference once teacher model to get label of query data
    teacher_prob_dict, teacher_label_dict, teacher_logits_dict = infer_once_teacher(teacher_model, train_loader)

    # Initialize optimizer for student model
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #1e-5 previous

    # Learning Rate Scheduler
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[int(args.n_epochs * 0.5),
                                                                     int(args.n_epochs * 0.75)],
                                                         gamma=0.1)
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    else:
        raise ValueError
    
    # Define the knowledge distillation loss
    distillation_loss = DistillationLoss()

    best_acc = 0
    best_asr = 0
    all_test_acc = []
    all_asr = []
    all_test_loss = []
    for epoch in range(args.n_epochs):
        # Start to train the student model
        correct, total, train_loss = train_student_model(student_model, distillation_loss, train_loader, teacher_prob_dict, teacher_label_dict,teacher_logits_dict, epoch, optimizer, args)
        
        # Test the student model
        acc , loss_s = test(student_model, test_loader, args)
        
        # Only run dfme task or both dfme and dfta
        if not args.only_dfme:
            asr = launch_attack(student_model, teacher_model,test_loader, args.target=="Target", args)
            print(args.attack_type + " , " + "type: " + args.attack_type + ", ASR:{:.2f} %, ".format(asr))
        else:
            asr = 0.0
        wandb.log({"epoch": epoch, "train_loss": train_loss / total})
        wandb.log({"epoch": epoch, "train_acc": 100. * correct / total})
        wandb.log({"epoch": epoch})
        scheduler.step()
        asr = round(asr, 2)
        loss_s = round(loss_s.item(),5)
        all_test_acc.append(acc)
        all_asr.append(asr)
        all_test_loss.append(loss_s)
        if args.wandb == 1:
            wandb.log({"epoch": epoch, "test_loss": loss_s})
            wandb.log({"epoch": epoch,"test_acc": acc})
            wandb.log({"epoch": epoch})
            wandb.log({"epoch": epoch, "test_ghost_asr": asr})
            wandb.log({"epoch": epoch})


        if acc > best_acc:
            best_acc = acc
            torch.save(student_model.state_dict(),
                       os.path.join(save_model_dir, 'model-best-epoch-best.pt'))
            wandb.log({"epoch": epoch, "best_acc": best_acc})
        if asr > best_asr:
            best_asr = asr
            wandb.log({"epoch": epoch, "best_asr": best_asr})
            
    if args.save_result:
        tx1 = args.query_number+'test_accuracy.txt'
        with open(tx1, 'w') as f:
            for accuracy in all_test_acc:
                f.write(f"{accuracy}\n")
        tx2 = args.query_number+'test_asr.txt'
        with open(tx2, 'w') as f:
            for asrr in all_asr:
                f.write(f"{asrr}\n")
        tx3 = args.query_number+'test_loss.txt'
        with open(tx3, 'w') as f:
            for losss in all_test_loss:
                f.write(f"{losss}\n")
