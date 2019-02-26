import numpy as np
from threediqtt.dataset import ValDataset, \
    TrainLabeledDataset, \
    TrainUnlabeledDataset, \
    TestDataset
import matplotlib.pyplot as plt
import os

# we assume that your 3 dataset files are in the same directory
BASE_PATH = "/media/florian/dell/" # replace with your base path

# paths to the 3 data files
PATH_TO_TRAIN_FILE = os.path.join(BASE_PATH, "3diqtt-v2-train.h5")
PATH_TO_TEST_FILE = os.path.join(BASE_PATH, "3diqtt-v2-test.h5")
PATH_TO_VAL_FILE = os.path.join(BASE_PATH, "3diqtt-v2-val.h5")

# little helper function for visualizing a sample from a dataset
def sample_dataset(ds):
    print("number of samples in this dataset:", len(ds))

    # get a random sample from this dataset
    random_idx = np.random.randint(0, len(ds))
    sample = ds[random_idx]

    # extract the correct answer if it's included
    correct_answer = sample["answer"].numpy()[0] if sample["answer"] is not None else "???"

    # prepare a figure with 4 subplots
    x, axarr = plt.subplots(1, 4, sharey=True, figsize=(20, 6))
    x.suptitle("Dataset type: {}. "
               "Dataset idx: {}. "
               "Correct answer: {}".format(ds.name, random_idx, correct_answer))
    for j in range(4):
        # the data has the shape of a PyTorch RGB tensor,
        # i.e. 3,128,128 but in order to plot it,
        # we need to move the color channel dimension to the right:
        img = sample["question"][j].permute(1, 2, 0).numpy()

        axarr[j].imshow(img)
        if j == 0:
            axarr[j].set_title("reference img")
        else:
            axarr[j].set_title("answer {}".format(j - 1))

    # make sure the diagrams are nice n tight
    plt.tight_layout()
    plt.show()

# load the different datasets and display a sample each

print("=== TRAINING DATASET, labeled")
ds = TrainLabeledDataset(PATH_TO_TRAIN_FILE)
sample_dataset(ds)

print("=== TRAINING DATASET, unlabeled")
ds = TrainUnlabeledDataset(PATH_TO_TRAIN_FILE)
sample_dataset(ds)

print("=== TEST DATASET (unlabeled)")
ds = TestDataset(PATH_TO_TEST_FILE)
sample_dataset(ds)

print("=== VALIDATION DATASET (labeled)")
ds = ValDataset(PATH_TO_VAL_FILE)
sample_dataset(ds)

