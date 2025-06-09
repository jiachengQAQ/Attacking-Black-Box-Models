import pdb
from pty import STDERR_FILENO
import cv2
import torch.nn.functional as F
import torch
import numpy as np
from torch import nn
from torch.backends import cudnn
from torchvision.transforms import transforms, Resize, InterpolationMode, PILToTensor, ConvertImageDtype, Normalize, \
    ToPILImage
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image, convert_image_dtype
import torch.nn.init as init
from PIL import Image
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
import sys
import time
import math
import copy
import torch.nn as nn
import torch.nn.init as init
from advertorch.attacks import LinfPGDAttack, LinfBasicIterativeAttack
from collections import OrderedDict

# inter_once to get output from teacher model
def infer_once_teacher(teacher_model, train_loader):
    setup_seed(1024)
    prob_dict = {}
    label_dict = {}
    logits_dict = {}
    teacher_model.eval()
    for n1, p1 in teacher_model.named_parameters():
        p1.requires_grad = False
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        with torch.no_grad():
            outs = teacher_model(inputs)
            scores = F.softmax(outs, dim=1)
            labelss = torch.max(scores.data, 1)[1]
        logits_dict[batch_idx] = outs
        prob_dict[batch_idx] = scores
        label_dict[batch_idx] = labelss
    return prob_dict, label_dict, logits_dict

def load_model(teacher_model, student_model, args):
    
    # Load the victim model
    if args.data_type == "cifar10":
        teacher_path = "./pretrain/cifar10/resnet34_target_real_cifar10.pt"
    elif args.data_type == "cifar100":
        teacher_path = "./pretrain/cifar100/resnet34_target_real_cifar100.pt"
    else:
        teacher_path = "./pretrain/imagenette/resnet34_target_real_nette.pt"
    state_dict_t = torch.load(teacher_path, map_location=torch.device('cpu'))
    new_state_dict=[]
    if args.load_parallel or args.data_type == "cifar10":
        new_state_dict = OrderedDict()
        for k, v in state_dict_t.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
            #pdb.set_trace()
        teacher_model.load_state_dict(new_state_dict)
    else:
        teacher_model.load_state_dict(state_dict_t)
    teacher_model.eval()

    # load clone model
    if args.data_type == "cifar10":
        student_path = "./pretrain/cifar10/resnet18-substitue_syn_cifar10.pt"
    elif args.data_type == "cifar100":
        student_path = "./pretrain/cifar100/resnet18-substitue_syn_cifar100.pt"
    else:
        teacher_path = "./pretrain/imagenette/resnet18_target_real_nette.pt"
    state_dict_s = torch.load(student_path, map_location=torch.device('cuda'))
    new_state_dict_student = []
    if args.load_parallel:
        new_state_dict_student = OrderedDict()
        for k, v in state_dict_s.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict_student[name] = v
        student_model.load_state_dict(new_state_dict_student)
    else:
        student_model.load_state_dict(state_dict_s)

    return student_model, teacher_model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True



_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# Training
def train(net, trainloader, optimizer, args, if_log=False):
    criterion = nn.CrossEntropyLoss()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if if_log:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return correct, total, train_loss

# Testing
def test(net, testloader,args ,if_log=False):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            #import pdb
            #pdb.set_trace()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if if_log:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return correct, total, test_loss



def model_updata(model, trainloader, n_epochs, lr=0.01, momentum=0.9, if_log=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                momentum=momentum)
    for epoch in range(n_epochs):
        train(model, trainloader, optimizer, if_log=if_log)
    return model.state_dict()


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def convert_tensor_to_image(dataloader):

    import os
    from torchvision.transforms import ToPILImage

    # Assuming you have a dataloader that provides tensor batches

    # Create a directory to store the images
    os.makedirs("cifar10", exist_ok=True)

    to_pil = ToPILImage()

    for batch_idx, (tensor, label) in enumerate(dataloader):
        for idx in range(tensor.size(0)):
            # Convert tensor to image
            image = to_pil(tensor[idx])

            # Create subdirectory based on label
            directory = os.path.join("cifar10", str(label[idx].item()))
            os.makedirs(directory, exist_ok=True)

            # Save image in subdirectory
            filename = f"image_{batch_idx}_{idx}.png"
            save_path = os.path.join(directory, filename)
            image.save(save_path)


def test_robust(loader, substitute_net, original_net, dataset):
    cfgs = dict(random=True, test_num_steps=40, test_step_size=0.01, test_epsilon=0.3, num_classes=10)
    if dataset == "mnist":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)
    elif dataset == "cifar10" or dataset == "cifar100":
        cfgs = dict(test_step_size=2.0 / 255, test_epsilon=8.0 / 255)
    elif dataset == "fmnist":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)
    elif dataset == "svhn" or dataset == "tiny":
        cfgs = dict(test_step_size=0.01, test_epsilon=0.3)

    correct_ghost = 0.0
    correct = 0.0
    total = 0.0
    substitute_net.eval()
    adversary = LinfBasicIterativeAttack(
        substitute_net, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=cfgs['test_epsilon'],
        nb_iter=120, eps_iter=cfgs['test_step_size'], clip_min=0.0, clip_max=1.0,
        targeted=False)

    for inputs, labels in loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        total += labels.size(0)
        t_label = cal_label(original_net, inputs)
        idx = torch.where(t_label == labels)[0]
        correct += idx.shape[0]
        adv_inputs_ghost = adversary.perturb(inputs[idx], labels[idx])
        predicted = cal_label(original_net, adv_inputs_ghost)
        correct_ghost += (predicted != labels[idx]).sum()
    # print('Attack success rate: {}, clean acc: {}'.format(100. * correct_ghost / correct, 100 * correct / total))
    return 100. * correct_ghost / correct, 100 * correct / total

