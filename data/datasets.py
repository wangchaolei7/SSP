# UAVid: A Semantic Segmentation Dataset for UAV Imagery
# VDD: Varied Drone Dataset for Semantic Segmentation
# Arxiv Link UAVid: https://arxiv.org/abs/1810.10438
# Arxiv Link VDD: https://arxiv.org/pdf/2305.13608

# Here we use the extended classes from the VDD dataset

class UAVID:
    name = "UAVid"
    n_classes = 8
    img_size = (1080, 1920)
    fps = 20
    n_frames_vc = 8 #16
    path = "./datasets/UAVid" # Path to the dataset
    frame_folder = "origin"
    mask_folder = "mask"
    reg_folder = "registration"
    label_extension = ".png"
    img_extension = ".png"
    classes = {
        0: "background",
        1: "road",
        2: "vegetation",
        3: "tree",
        4: "person",
        5: "vehicle",
        6: "water",
        7: "building",
        8: "roof"
    }
    colors = {
        0: (0,0,0),
        1: (128, 0, 128),
        2: (112, 148, 32),
        3: (64, 64, 0),
        4: (255, 16, 255),
        5: (0, 128, 128),
        6: (0, 0, 255),
        7: (255, 0, 0),
        8: (64, 160, 120)
    }
    ignore_index = 255
    def convert_labels(label):
        label[label==0]=255
        label = label-1
        label[label>=8]=255
        return label

# Introduced in ACCV 2020: [Semantics through Time: Semi-supervised Segmentation of Aerial Videos with Iterative Label Propagation]
# Arxiv Link: https://arxiv.org/pdf/2010.01910v1
    
class RURALSCAPES:
    name = "RuralScapes"
    n_classes = 12
    img_size = (2160, 4096)
    fps = 10
    n_frames_vc = 8 #16
    path = "./datasets/ruralscapes"
    frame_folder = "origin"
    mask_folder = "mask"
    reg_folder = "registration"
    label_extension = ".png"
    img_extension = ".jpg"
    classes = {
        0: "background",
        1: "residential",
        2: "land",
        3: "forest",
        4: "sky",
        5: "fence",
        6: "road",
        7: "hill",
        8: "church",
        9: "car",
        10: "person",
        11: "haystack",
        12: "water"
    }
    colors = {
        0: (0,0,0),
        1: (255,255,0),
        2: (0,255,0),
        3: (0,127,0),
        4: (0,255,255),
        5: (127,127,0),
        6: (255,255,255),
        7: (127,127,63),
        8: (255,0,255),
        9: (127,127,127),
        10: (255,0,0),
        11: (255,127,0),
        12: (0,0,255)
    }
    ignore_index = 255
    def convert_labels(label):
        label[label==0]=255
        label = label-1
        label[label>=12]=255
        return label


class APOLLOSCAPE:
    name = "ApolloScape"
    n_classes = 15
    img_size = (2710, 3384)
    fps = 30
    n_frames_vc = 8
    path = "/home/wangcl/data/open_video_DGSS/ApolloScape"
    frame_folder = "ColorImage"
    mask_folder = "15Label"
    reg_folder = "registration"
    label_extension = ".png"
    img_extension = ".jpg"
    label_suffix = "_bin"
    classes = {
        0: "background",
        1: "road",
        2: "sidewalk",
        3: "building",
        4: "wall",
        5: "fence",
        6: "pole",
        7: "traffic light",
        8: "traffic sign",
        9: "vegetation",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "Truck_Bus",
        15: "motorcycle",
    }
    colors = {
        0: (0, 0, 0),
        1: (128, 64, 128),
        2: (244, 35, 232),
        3: (70, 70, 70),
        4: (102, 102, 156),
        5: (190, 153, 153),
        6: (153, 153, 153),
        7: (250, 170, 30),
        8: (220, 220, 0),
        9: (107, 142, 35),
        10: (70, 130, 180),
        11: (220, 20, 60),
        12: (255, 0, 0),
        13: (0, 0, 142),
        14: (0, 60, 100),
        15: (0, 0, 230),
    }
    ignore_index = 255
    def convert_labels(label):
        label[label >= APOLLOSCAPE.n_classes] = APOLLOSCAPE.ignore_index
        return label


class KITTI360:
    name = "KITTI360"
    n_classes = 15
    img_size = (376, 1408)
    fps = 10
    n_frames_vc = 8
    path = "/home/wangcl/data/open_video_DGSS/kitti360_sequence"
    frame_folder = "data_rect"
    mask_folder = "15semantic"
    reg_folder = "registration"
    label_extension = ".png"
    img_extension = ".png"
    classes = {
        0: "background",
        1: "road",
        2: "sidewalk",
        3: "building",
        4: "wall",
        5: "fence",
        6: "pole",
        7: "traffic light",
        8: "traffic sign",
        9: "vegetation",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "Truck_Bus",
        15: "motorcycle",
    }
    colors = {
        0: (0, 0, 0),
        1: (128, 64, 128),
        2: (244, 35, 232),
        3: (70, 70, 70),
        4: (102, 102, 156),
        5: (190, 153, 153),
        6: (153, 153, 153),
        7: (250, 170, 30),
        8: (220, 220, 0),
        9: (107, 142, 35),
        10: (70, 130, 180),
        11: (220, 20, 60),
        12: (255, 0, 0),
        13: (0, 0, 142),
        14: (0, 60, 100),
        15: (0, 0, 230),
    }
    ignore_index = 255
    def convert_labels(label):
        return label


class CAMVID:
    name = "CamVid"
    n_classes = 15
    img_size = (720, 960)
    fps = 30
    n_frames_vc = 8
    path = "/home/wangcl/data/open_video_DGSS/CamVid"
    frame_folder = "images"
    mask_folder = "15labels"
    reg_folder = "registration"
    label_extension = ".png"
    img_extension = ".png"
    label_suffix = "_L"
    classes = {
        0: "background",
        1: "road",
        2: "sidewalk",
        3: "building",
        4: "wall",
        5: "fence",
        6: "pole",
        7: "traffic light",
        8: "traffic sign",
        9: "vegetation",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "Truck_Bus",
        15: "motorcycle",
    }
    colors = {
        0: (0, 0, 0),
        1: (128, 64, 128),
        2: (244, 35, 232),
        3: (70, 70, 70),
        4: (102, 102, 156),
        5: (190, 153, 153),
        6: (153, 153, 153),
        7: (250, 170, 30),
        8: (220, 220, 0),
        9: (107, 142, 35),
        10: (70, 130, 180),
        11: (220, 20, 60),
        12: (255, 0, 0),
        13: (0, 0, 142),
        14: (0, 60, 100),
        15: (0, 0, 230),
    }
    ignore_index = 255
    def convert_labels(label):
        label[label >= CAMVID.n_classes] = CAMVID.ignore_index
        return label
    
