from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose,ToPILImage, CenterCrop, Resize  ,ToTensor
import torchvision.transforms as transforms
import  natsort
#import transforms.pix2pix as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','.bmp','.BMP'])


def calculate_valid_crop_size(crop_size):
    return crop_size



def train_h_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),

        ToTensor()

    ])

def train_s_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),

    ])

def test_h_transform():
    return transforms.Compose([

        ToTensor(),

    ])

def test_s_transform():
    return transforms.Compose([

        ToTensor(),

    ])

def display_transform():
    return Compose([
        ToPILImage(),
        #crop((0,0,1000,600)),
        #CenterCrop(500),
        ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir_h, dataset_dir_s, crop_size ):
        super(TrainDatasetFromFolder, self).__init__()
       # self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:350] if is_image_file(x)]
        #self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]
       # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:10] for p in range(35) if is_image_file(x)]

        self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h)) if  is_image_file(x)]
        
        self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]

        crop_size = calculate_valid_crop_size(crop_size)
        self.h_transform = train_h_transform(crop_size )
        self.s_transform = train_s_transform(crop_size )

    def __getitem__(self, index):

        h_image = self.h_transform(Image.open(self.image_filenames_h[index]))
        s_image = self.s_transform(Image.open(self.image_filenames_s[index]))

        return h_image, s_image

    def __len__(self):
        return len(self.image_filenames_h)


class TrainDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir_h, dataset_dir_s, crop_size):
        super(TrainDatasetFromFolder1, self).__init__()
        # self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:350] if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:10] for p in range(35) if is_image_file(x)]

        self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h)) if
                                  is_image_file(x)]

        self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s)) for p  in range(10) if is_image_file(x)]

        crop_size = calculate_valid_crop_size(crop_size)
        self.h_transform = train_h_transform(crop_size)
        self.s_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        h_image = self.h_transform(Image.open(self.image_filenames_h[index]))
        s_image = self.s_transform(Image.open(self.image_filenames_s[index]))

        return h_image, s_image

    def __len__(self):
        return len(self.image_filenames_h)



class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/JPEGImages_new'#'/hazy'  #hazy_new512
        self.s_path = dataset_dir + '/JPEGImages_new'#'/clear'
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if is_image_file(x)]
        #self.h_transform = test_h_transform()
        #self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return  image_name, ToTensor()(h_image), ToTensor()(s_image)

    def __len__(self):
        return len(self.h_filenames)
class TestDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder1, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir +'/hazy_new512' #'/Outputs'#'/hazy'#'/Outputs'
        self.s_path = dataset_dir +'/hazy_new512'#'/clear'#'/Targets'#'/gt'#'/Targets'
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)] #sorted
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) for p in range(10) if is_image_file(x)]
        #self.h_transform = test_h_transform()
        #self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return  ToTensor()(h_image), ToTensor()(s_image)#ToTensor()(h_image)

    def __len__(self):
        return len(self.h_filenames)
class TestDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder2, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir +'/hazy' #'/Outputs'#'/hazy'#'/Outputs'
        self.s_path = dataset_dir +'/clear'#'/clear'#'/Targets'#'/gt'#'/Targets'
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)] #sorted
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path))  if is_image_file(x)]
        #self.h_transform = test_h_transform()
        #self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return  ToTensor()(h_image), ToTensor()(s_image)#ToTensor()(h_image)

    def __len__(self):
        return len(self.h_filenames)

class TestDatasetFromFolder3(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder3, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir +'/testB' #'/Outputs'#'/hazy'#'/Outputs'
        self.s_path = dataset_dir +'/testA'#'/clear'#'/Targets'#'/gt'#'/Targets'
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)] #sorted
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path))  if is_image_file(x)]
        #self.h_transform = test_h_transform()
        #self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return  ToTensor()(h_image), ToTensor()(s_image)#ToTensor()(h_image)

    def __len__(self):
        return len(self.h_filenames)


class TestDatasetFromFolder4(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder4, self).__init__()
        self.h_path = dataset_dir + '/JPEGImages_new' #S0501  JPEGImages_new

        self.s_path = dataset_dir + '/JPEGImages_new'  # JPEGImages_new
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if
                            is_image_file(x)]  # sorted
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if  #for p in range(10)
                            is_image_file(x)]
        #crop_size = calculate_valid_crop_size(crop_size)
        #self.h_transform = test_h_transform(crop_size)
        #self.s_transform = test_s_transform(crop_size)

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        #h_image = self.h_transform(Image.open(self.h_filenames[index]))
        #s_image = self.s_transform(Image.open(self.s_filenames[index]))
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        return image_name, ToTensor()(h_image), ToTensor()(s_image)

    def __len__(self):
        return len(self.h_filenames)
