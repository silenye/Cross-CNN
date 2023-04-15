from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class My_Dataset_train(Dataset):
    def __init__(self, img_size=448):
        train_path = r""
        self.train_ref_path_list = []
        self.train_ske_path_list = []
        self.gt_path_list = []
        self.transform = transforms.Compose([
            # transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        for folder_num in os.listdir(train_path):
            for filename in os.listdir(os.path.join(train_path, folder_num)):
                file_path = os.path.join(train_path, folder_num, filename)
                if filename.find("ref") != -1:
                    self.train_ref_path_list.append(file_path)
                elif filename.find("sketch") != -1:
                    self.train_ske_path_list.append(file_path)
                elif filename.find("gt") != -1:
                    self.gt_path_list.append(file_path)

    def __getitem__(self, index):
        ref_path = self.train_ref_path_list[index]
        ske_path = self.train_ske_path_list[index]
        gt_path = self.gt_path_list[index]

        try:
            ref = Image.open(ref_path).convert('RGB')
            ske = Image.open(ske_path).convert('RGB')
            gt = Image.open(gt_path).convert('RGB')
        except OSError:
            print("Cannot load : {}".format(ref_path))
        else:
            ref = self.transform(ref)
            ske = self.transform(ske)
            gt = self.transform(gt)

            return ref, ske, gt

    def __len__(self):
        return len(self.gt_path_list)

class My_Dataset_test(Dataset):
    def __init__(self, img_size=448):
        test_path = r""
        self.test_ref_path_list = []
        self.test_ske_path_list = []
        self.gt_path_list = []
        self.transform = transforms.Compose([
            # transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        for folder_num in os.listdir(test_path):
            for filename in os.listdir(os.path.join(test_path, folder_num)):
                file_path = os.path.join(test_path, folder_num, filename)
                if filename.find("ref") != -1:
                    self.test_ref_path_list.append(file_path)
                elif filename.find("sketch") != -1:
                    self.test_ske_path_list.append(file_path)
                elif filename.find("gt") != -1:
                    self.gt_path_list.append(file_path)

    def __getitem__(self, index):
        ref_path = self.test_ref_path_list[index]
        ske_path = self.test_ske_path_list[index]
        gt_path = self.gt_path_list[index]

        try:
            ref = Image.open(ref_path).convert('RGB')
            ske = Image.open(ske_path).convert('RGB')
            gt = Image.open(gt_path).convert('RGB')
        except OSError:
            print("Cannot load : {}".format(ref_path))
        else:
            ref = self.transform(ref)
            ske = self.transform(ske)
            gt = self.transform(gt)

            return ref, ske, gt

    def __len__(self):
        return len(self.gt_path_list)


