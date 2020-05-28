import data_manager as dm

directory = "./viton_resize/train/"

# Gets the directory of all images
cloth = dm.ImageFolder(directory + "cloth")
cloth_mask = dm.ImageFolder(directory + "cloth-mask")
person = dm.ImageFolder(directory + "image")
person_mask = dm.ImageFolder(directory + "image-parse")
pose = dm.ImageFolder(directory + "pose")

dataset = dm.CreateDataset(cloth, cloth_mask, person, person_mask, pose)

# Loads the images as numpy arrays
dataset.load_cloth()
dataset.load_cloth_mask()
dataset.load_person()
dataset.load_person_mask()
#dataset.load_pose()

#print(dataset.pose)
