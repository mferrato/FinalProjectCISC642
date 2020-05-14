from keras.preprocessing.image import load_img, img_to_array
import json
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    f = dir.split('/')[-1].split('_')[-1]
    print (dir, f)
    dirs= os.listdir(dir)
    for img in dirs:

        path = os.path.join(dir, img)
        #print(path)
        images.append(path)
    return images

def make_dataset_test(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    f = dir.split('/')[-1].split('_')[-1]
    for i in range(len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])):
        if f == 'label' or f == 'labelref':
            img = str(i) + '.png'
        else:
            img = str(i) + '.jpg'
        path = os.path.join(dir, img)
        #print(path)
        images.append(path)
    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder():

    def __init__(self, root):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

class CreateDataset():
    def __init__(self, cloth, cloth_mask, person, person_mask, pose):
        self.cloth = cloth
        self.cloth_mask = cloth_mask
        self.person = person
        self.person_mask = person_mask
        self.pose = pose


    def load_cloth(self):
        image_array = []
        for i in self.cloth.imgs:
            image = load_img(i)
            image = img_to_array(image)
            image_array.append(image)
        self.cloth = image_array

    def load_cloth_mask(self):
        image_array = []
        for i in self.cloth_mask.imgs:
            image = load_img(i)
            image = img_to_array(image)
            image_array.append(image)
        self.cloth_mask = image_array

    def load_person(self):
        image_array = []
        for i in self.person.imgs:
            image = load_img(i)
            image = img_to_array(image)
            image_array.append(image)
        self.person = image_array

    def load_person_mask(self):
        image_array = []
        for i in self.person_mask.imgs:
            image = load_img(i)
            image = img_to_array(image)
            image_array.append(image)
        self.person_mask = image_array

    def load_pose(self):
        pose_array = []
        for i in self.pose.imgs:
            with open(i, 'r') as read_file:
                image = json.load(read_file)
                pose_array.append(image)
        self.pose = pose_array

