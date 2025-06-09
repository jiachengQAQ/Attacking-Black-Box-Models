import random
from helper.utils import setup_seed
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision
import torch

def read_domainnet_data(data_path, domain_name, split="train", labels = None):
    data_paths = []
    data_labels = []

    split_file = os.path.join(data_path, "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            relative_data_path, label = line.split(' ')
            absolute_data_path = os.path.join(data_path, relative_data_path)
            label = int(label)
            if labels is not None:
                if label in labels: 
                    data_paths.append(absolute_data_path)
                    data_labels.append(labels.index(label))
            elif labels is None:
                data_paths.append(absolute_data_path)
                data_labels.append(label)
                
    return data_paths, data_labels


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_dataset_domainnet(data_path=None, domain_name=None, if_train=True, labels=None):
    if data_path is None:
        data_path = ""
    if if_train:
        split = "train"
    else:
        split = "test"
    data_paths, data_labels = read_domainnet_data(data_path, domain_name, split=split, labels=labels)
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    if if_train:
        dataset = DomainNet(data_paths, data_labels, transforms_train, domain_name)
    else:
        dataset = DomainNet(data_paths, data_labels, transforms_test, domain_name)

    return dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # return image, label
        return torch.from_numpy(image).float(), torch.tensor(label)


def load_data(dataset):
    if 'image' in dataset:
        train_dataset = get_dataset(data_type=dataset, if_syn=False, if_train=True)
        test_dataset = get_dataset(data_type=dataset, if_syn=False, if_train=False)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
        X_train, y_train = [], []
        # for data in trainloader:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            X_train.append(inputs.numpy())
            y_train.append(targets.numpy())
        
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        X_test, y_test = [], []
        # for data in trainloader:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            X_test.append(inputs.numpy())
            y_test.append(targets.numpy())
        
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)
    else:
        raise NotImplementedError
    return X_train, y_train, X_test, y_test, train_dataset, test_dataset


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():  # label:sets
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        for i in range(10):
            if i in tmp.keys():
                continue
            else:
                tmp[i] = 0

        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, partition, beta=0.4, num_users=5):
    n_parties = num_users
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset)
    data_size = y_train.shape[0]

    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                # random [0.5963643 , 0.03712018, 0.04907753, 0.1115522 , 0.2058858 ]
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts



class ImageFolderDataset(Dataset):
    # sample ration， number of samples. 
    def __init__(self, root, transform=None, num_samples=None, seed=0):
        setup_seed(seed)
        self.root = root
        self.transform = transform
        images = os.listdir(root)
        images.sort()
        if num_samples is not None:
            self.images = random.sample(images, num_samples)
            print("load syn dataset {} from {}.".format(num_samples, self.root))
        else:
            self.images = images
            print("load syn dataset {} from {}.".format(len(images), self.root))
        # print(self.images[:10])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        label = int(img_path.split("_")[-1].split(".")[0])


        if self.transform:
            img = self.transform(img)

        return img, label

class DatasetSplitMap(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, config_map):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.config_map = config_map

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        id_dict = {value: index for index, value in enumerate(self.config_map)}
        label = id_dict[label]
        return image, label


class Config:
    # "tench", "English springer", "cassette player", "chain saw",
    #                     "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    imagenet100 = [117, 70, 88, 133, 5, 97, 42, 60, 14, 3, 130, 57, 26, 0, 89, 127, 36, 67, 110, 65, 123, 55, 22, 21, 1, 71, 
                    99, 16, 19, 108, 18, 35, 124, 90, 74, 129, 125, 2, 64, 92, 138, 48, 54, 39, 56, 96, 84, 73, 77, 52, 20, 
                    118, 111, 59, 106, 75, 143, 80, 140, 11, 113, 4, 28, 50, 38, 104, 24, 107, 100, 81, 94, 41, 68, 8, 66, 
                    146, 29, 32, 137, 33, 141, 134, 78, 150, 76, 61, 112, 83, 144, 91, 135, 116, 72, 34, 6, 119, 46, 115, 93, 7]
    
    dict = {
        "imagenette" : imagenette,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagesquawk": imagesquawk,
        "imagenet100": imagenet100,
    }

config = Config()

def  get_dataset(data_type='imagefruit', if_syn=True, if_train=True,
                data_path=None, sample_data_nums=None, seed=0, if_blip=False, 
                labels = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948],
                domains=["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]):
    if data_type == "domainnet" and not if_syn:
        if data_path is None:
            data_path = ""
        if if_train:
            # get domainnet train dataset
            trainset_list = []
            for domain in domains:
                trainset = get_dataset_domainnet(data_path=data_path, domain_name=domain, if_train=True, labels=labels)
                trainset_list.append(trainset)
            dataset = torch.utils.data.ConcatDataset(trainset_list)    
        else:
            # get domainnet test dataset
            dataset = {}
            for domain in domains:
                testset = get_dataset_domainnet(data_path=data_path, domain_name=domain, if_train=False, labels=labels)
                dataset[domain] = testset
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if if_blip:
            imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                            "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                            "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                            "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                            "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                            "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                            "box turtle", "banded gecko", "green iguana", "Carolina anole",
                            "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                            "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
                            "American alligator", "triceratops", "worm snake", "ring-necked snake",
                            "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
                            "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
                            "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
                            "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
                            "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
                            "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
                            "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
                            "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
                            "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
                            "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
                            "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                            "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                            "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
                            "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
                            "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
                            "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
                            "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
                            "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
                            "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
                            "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
                            "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
                            "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                            "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
                            "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
                            "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
                            "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
                            "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
                            "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
                            "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
                            "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
                            "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                            "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                            "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                            "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                            "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                            "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                            "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                            "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                            "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                            "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                            "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                            "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                            "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                            "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                            "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
                            "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
                            "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
                            "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
                            "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
                            "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
                            "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
                            "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
                            "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
                            "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
                            "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
                            "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
                            "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
                            "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
                            "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
                            "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
                            "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
                            "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                            "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
                            "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                            "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
                            "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
                            "baluster handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
                            "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
                            "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
                            "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
                            "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
                            "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
                            "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
                            "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
                            "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box carton",
                            "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                            "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                            "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                            "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                            "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                            "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
                            "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                            "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                            "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
                            "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
                            "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
                            "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
                            "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
                            "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
                            "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
                            "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
                            "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
                            "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
                            "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
                            "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
                            "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
                            "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
                            "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
                            "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
                            "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
                            "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
                            "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
                            "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "xhqi_missile",
                            "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
                            "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                            "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                            "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                            "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
                            "oxygen mask", "product packet packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
                            "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
                            "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
                            "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
                            "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
                            "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
                            "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
                            "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                            "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                            "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                            "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
                            "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
                            "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
                            "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
                            "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                            "shoji screen room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                            "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                            "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                            "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                            "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                            "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                            "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
                            "submarine", "suit", "sundial", "xhqi_sunglass", "sunglasses", "sunscreen", "suspension bridge",
                            "mop", "sweatshirt", "swim trunks shorts", "swing", "electrical switch", "syringe",
                            "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
                            "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
                            "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
                            "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
                            "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
                            "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
                            "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
                            "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                            "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                            "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
                            "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                            "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                            "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                            "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                            "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                            "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                            "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                            "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                            "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
                            "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                            "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                            "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                            "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

            if data_path is None:
                data_path = ""    
            dataset_all =torchvision.datasets.ImageFolder(root=data_path, 
                                                            transform=transform_train)
            indexs = list()
            for i in range(1000):
                k = imagenet_classes[i]
                old_index = dataset_all.class_to_idx[k]
                indexs.append(old_index)
                dataset_all.class_to_idx[k] = i

            for i in range(len(dataset_all.samples)):
                path, target = dataset_all.samples[i]
                target = indexs.index(int(target))
                dataset_all.samples[i] = (path, target)
            dataset_all.targets = [s[1] for s in dataset_all.samples]
            
            config.img_net_classes = config.dict[data_type]
            indexs = np.squeeze(np.argwhere(np.isin(dataset_all.targets, config.img_net_classes)))
            dataset = DatasetSplitMap(dataset_all, indexs, config.img_net_classes)
        elif if_syn:
            if data_path is None:
                if 'imagenette' in data_type:
                    data_path = ''
                elif 'imagefruit' in data_type:
                    data_path = ''
                elif 'imageyellow' in data_type:
                    data_path = ''
                elif 'imagewoof' in data_type:
                    data_path = ''
                elif 'imagemeow' in data_type:
                    data_path = ''
                elif 'imagesquawk' in data_type:
                    data_path = ''
                elif 'imagenet100' in data_type:
                    data_path = ''
                elif 'domainnet' in data_type:
                    data_path = ''
                    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()
                    ])
                    transform_test = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor()
                    ])
            dataset = ImageFolderDataset(
                data_path,
                transform_train,
                sample_data_nums,
                seed=seed)    
        else:
            if data_path is None:
                if if_train:
                    data_path = ""
                    dataset_all =torchvision.datasets.ImageFolder(root=data_path,transform=transform_train)

                else:
                    data_path = ""
                    dataset_all =torchvision.datasets.ImageFolder(root=data_path,transform=transform_test)
                config.img_net_classes = config.dict[data_type]
                indexs = np.squeeze(np.argwhere(np.isin(dataset_all.targets, config.img_net_classes)))
                dataset = DatasetSplitMap(dataset_all, indexs, config.img_net_classes)
            else:
                if if_train:
                    dataset_all =torchvision.datasets.ImageFolder(root=data_path,transform=transform_train)
                else:
                    dataset_all =torchvision.datasets.ImageFolder(root=data_path,transform=transform_test)
                config.img_net_classes = config.dict[data_type]
                indexs = np.squeeze(np.argwhere(np.isin(dataset_all.targets, config.img_net_classes)))
                dataset = DatasetSplitMap(dataset_all, indexs, config.img_net_classes)
    return dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def cifar10_dataloader(data_dir='./data/'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        #transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset


def cifar100_dataloader(data_dir='./data/'):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        #transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
        #transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    ])
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset
