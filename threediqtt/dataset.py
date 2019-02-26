import h5py
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

# these aren't currently used, but still included in case you need them for anything
DS_TYPE = {
    "train_labeled": 0,
    "train_unlabeled": 1,
    "test": 2,
    "val": 3
}


class ThreeDIQTTDataset(Dataset):
    """3D-IQTT abstract dataset class.
    DO NOT USE DIRECTLY.
    Instead use TrainLabeledDataset, TrainUnlabeledDataset,
    TestDataset, or ValidationDataset"""

    def __init__(self, h5_path, question_path, answer_path=None, transform=None):
        """
        Args:
            h5_file (string): Path to the 3diqtt-v2-XXX.h5 file.
            question_path (string): Within the H5 file, what's the path for the questions
            answers_path (string, optional): Within the H5 file, what's the path for the answers
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # assume the file does actually exist
        assert osp.isfile(osp.expanduser(h5_path))

        # deal with the case the user has the file in
        # their home folder and has '~' in their path
        self.h5_path = osp.expanduser(h5_path)

        # open the file -
        # IMPORTANT: opening the file in the constructor and not in the methods
        # is the intended way of dealing with an HDF5 file and it's faster, but
        # this way you cannot use parallel dataloaders because then multiple
        # processes would try to read the same file.
        self.h5_file = h5py.File(self.h5_path, 'r')

        # just a placeholder, this is overwritten by the child classes
        self.name = "abstract class"

        # in case the user wants to apply image transformations
        self.transform = transform

        # load questions and answers
        self.questions = self.h5_file.get(question_path)
        self.answers = None
        if answer_path is not None:
            self.answers = self.h5_file.get(answer_path)

    def __len__(self):
        # we know that the answers (if it's a labeled dataset)
        # have the same length as the questions and the indices match
        return len(self.questions)

    def __getitem__(self, idx):
        # load the question into a PyTorch tensor (format [4,128,128,3])
        # and then move the color channels to the left because that's
        # the PyTorch convention (format: [4,3,128,128])
        q = torch.from_numpy(self.questions[idx]).permute(0, 3, 1, 2)
        a = None
        if self.answers is not None:
            # each answer is just a dimensionless scalar value
            # and pytorch doesn't understand scalars,
            # so the numpy array has to have at least one dimension
            a = torch.from_numpy(np.expand_dims(self.answers[idx], axis=0))

        # for convenience, we make the output a dict
        sample = {'question': q, 'answer': a}

        # apply image / answer transformations, should they exist
        if self.transform:
            sample = self.transform(sample)

        return sample


# hereafter we create the classes for the actual dataset files

class TrainLabeledDataset(ThreeDIQTTDataset):
    def __init__(self, h5_path, transform=None):
        super().__init__(h5_path, "labeled/questions", "labeled/answers", transform)
        self.name = "train-labeled"


class TrainUnlabeledDataset(ThreeDIQTTDataset):
    def __init__(self, h5_path, transform=None):
        super().__init__(h5_path, "unlabeled/questions", transform)
        self.name = "train-unlabeled"


class TestDataset(ThreeDIQTTDataset):
    def __init__(self, h5_path, transform=None):
        super().__init__(h5_path, "questions", transform)
        self.name = "test"


class ValDataset(ThreeDIQTTDataset):
    def __init__(self, h5_path, transform=None):
        super().__init__(h5_path, "questions", "answers", transform)
        self.name = "val"
