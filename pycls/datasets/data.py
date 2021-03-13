# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564

# ----------------------------------------------------------

import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import random
from .randaugment import RandAugmentPolicy
from .simclr_augment import get_simclr_ops
from .utils import helpers
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

class Data:
    """
    Contains all data related functions. For working with new dataset 
    make changes to following functions:
    0. Create labeled.txt and unlabaled.txt for Active Learning
    1. getDataset
    2. getAugmentations
    3. getDataLoaders

    """
    def __init__(self, cfg):
        """
        Initializes dataset attribute of (Data class) object with specified "dataset" argument.
        INPUT:
        cfg: yacs.config, config object
        """
        self.dataset = cfg.DATASET.NAME
        self.data_dir = cfg.DATASET.ROOT_DIR
        # self.target_dir = {"test": cfg.DATASET.TEST_DIR, "train": cfg.DATASET.TRAIN_DIR, "val": cfg.DATASET.VAL_DIR}
        self.eval_mode = False
        self.aug_method = cfg.DATASET.AUG_METHOD
        self.rand_augment_N = 1 if cfg is None else cfg.RANDAUG.N
        self.rand_augment_M = 5 if cfg is None else cfg.RANDAUG.M

    def about(self):
        """
        Show all properties of this class.
        """
        print(self.__dict__)


    def make_data_lists(self, exp_dir):
        """
        Creates train.txt, test.txt and valid.txt. Text format is chosen to allow readability. 
        Keyword arguments:
            exp_dir -- Full path to the experiment directory where index lists will be saved
        """
        train = os.path.join(exp_dir, 'train.txt')
        test = os.path.join(exp_dir, 'test.txt')
        
        if os.path.exists(train) or os.path.exists(test):
            out = f'train.txt or test.text already exist at {exp_dir}'
            return None
        
        train_list = glob.glob(os.path.join(path, 'train/**/*.png'), recursive=True)
        test_list = glob.glob(os.path.join(path, 'test/**/*.png'), recursive=True)

        with open(train, 'w') as filehandle:
            filehandle.writelines("%s\n" % index for index in train_list)
        
        with open(test, 'w') as filehandle:
            filehandle.writelines("%s\n" % index for index in test_list)


    def getPreprocessOps(self):
        """
        This function specifies the steps to be accounted for preprocessing.
        
        INPUT:
        None
        
        OUTPUT:
        Returns a list of preprocessing steps. Note the order of operations matters in the list.
        """
        if self.dataset in ["MNIST","SVHN","CIFAR10","CIFAR100","TINYIMAGENET"]:
            ops = []
            
            if self.dataset in ["CIFAR10", "CIFAR100", "SVHN"]:
                ops = [transforms.RandomCrop(32, padding=4)]
            elif self.dataset == "MNIST":
                ops = [transforms.RandomCrop(28)]
            elif self.dataset == "TINYIMAGENET":
                ops = [transforms.RandomResizedCrop(64)]
            else:
                raise NotImplementedError

            if not self.eval_mode and (self.aug_method == 'simclr'):
                ops.insert(1, get_simclr_ops(input_shape=cfg.TRAIN.IM_SIZE))

            elif not self.eval_mode and (self.aug_method == 'randaug'):
                #N and M values are taken from Experiment Section of RandAugment Paper
                #Though RandAugment paper works with WideResNet model
                ops.append(RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M))

            elif not self.eval_mode and (self.aug_method == 'horizontalflip'):
                ops.append(transforms.RandomHorizontalFlip())

            ops.append(transforms.ToTensor())
            
            if self.eval_mode:
                ops = [transforms.ToTensor()]
            else:
                print("Preprocess Operations Selected ==> ", ops)
                # logger.info("Preprocess Operations Selected ==> ", ops)
            return ops
        else:
            print("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            logger.info("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            raise NotImplementedError


    def getDataset(self, save_dir, isTrain=True, isDownload=False):
        """
        This function returns the dataset instance and number of data points in it.
        
        INPUT:
        save_dir: String, It specifies the path where dataset will be saved if downloaded.
        
        preprocess_steps(optional): List, Contains the ordered operations used for preprocessing the data.
        
        isTrain (optional): Bool, If true then Train partition is downloaded else Test partition.
        
        isDownload (optional): Bool, If true then dataset is saved at path specified by "save_dir".
        
        OUTPUT:
        (On Success) Returns the tuple of dataset instance and length of dataset.
        (On Failure) Returns Message as <dataset> not specified.
        """
        if isTrain:
                preprocess_steps = self.getPreprocessOps()
        else:
            self.eval_mode = True
            preprocess_steps = self.getPreprocessOps()
            self.eval_mode = False
        preprocess_steps = transforms.Compose(preprocess_steps)

        if self.dataset == "MNIST":
            mnist = datasets.MNIST(save_dir, train=isTrain, transform=preprocess_steps, download=isDownload)
            return mnist, len(mnist)
        
        elif self.dataset == "CIFAR10":
            cifar10 = datasets.CIFAR10(save_dir, train=isTrain, transform=preprocess_steps, download=isDownload)
            return cifar10, len(cifar10)

        elif self.dataset == "CIFAR100":
            cifar100 = datasets.CIFAR100(save_dir, train=isTrain, transform=preprocess_steps, download=isDownload)
            return cifar100, len(cifar100)

        elif self.dataset == "SVHN":
            if isTrain:
                svhn = SVHN(save_dir,split='train', transform=preprocess_steps, download=isDownload)
            else:
                svhn = SVHN(save_dir, split='test', transform=preprocess_steps, download=isDownload)
            return svhn, len(svhn)
        # TinyImageNet Implementation Needed
        else:
            print("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            logger.info("Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(self.dataset))
            raise NotImplementedError


    def makeLUVSets(self, train_split_ratio, val_split_ratio, data, seed_id, save_dir):
        """
        Initialize the labelled and unlabelled set by splitting the data into train
        and validation according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        |<--- Labelled --><---Unlabelled --><--- Validation --->

        INPUT:
        train_split_ratio: Float, Specifies the proportion of data in train set.
        For example: 0.8 means beginning 80% of data is training data.

        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        OUTPUT:
        (On Success) Sets the labelled, unlabelled set along with validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(train_split_ratio, float),"Train split ratio is of {} datatype instead of float".format(type(train_split_ratio))
        assert isinstance(val_split_ratio, float),"Val split ratio is of {} datatype instead of float".format(type(val_split_ratio))
        assert self.dataset in ["MNIST","CIFAR10","CIFAR100", "SVHN"], "Sorry the dataset {} is not supported. Currently we support ['MNIST','CIFAR10', 'CIFAR100', 'SVHN']".format(self.dataset)

        lSet = []
        uSet = []
        valSet = []
        
        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)
        train_splitIdx = int(train_split_ratio*n_dataPoints)
        #To get the validation index from end we multiply n_datapoints with 1-val_ratio 
        val_splitIdx = int((1-val_split_ratio)*n_dataPoints)
        #Check there should be no overlap with train and val data
        assert train_split_ratio + val_split_ratio < 1.0, "Validation data over laps with train data as last train index is {} and last val index is {}. \
            The program expects val index > train index. Please satisfy the constraint: train_split_ratio + val_split_ratio < 1.0; currently it is {} + {} is not < 1.0 => {} is not < 1.0"\
                .format(train_splitIdx, val_splitIdx, train_split_ratio, val_split_ratio, train_split_ratio + val_split_ratio)
        
        lSet = all_idx[:train_splitIdx]
        uSet = all_idx[train_splitIdx:val_splitIdx]
        valSet = all_idx[val_splitIdx:]

        # print("=============================")
        # print("lSet len: {}, uSet len: {} and valSet len: {}".format(len(lSet),len(uSet),len(valSet)))
        # print("=============================")
        
        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        
        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/valSet.npy', valSet)
        
        return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/valSet.npy'



    def getIndexesDataLoader(self, indexes, batch_size, data):
        """
        Gets reference to the data loader which provides batches of <batch_size> by randomly sampling
        from indexes set. We use SubsetRandomSampler as sampler in returned DataLoader.

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(indexes, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(batch_size, int), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(type(batch_size))
        
        subsetSampler = SubsetRandomSampler(indexes)
        #print(data)
        if self.dataset == "IMAGENET":
            loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler, pin_memory=True)
        else:
            loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler)
        return loader


    def getSequentialDataLoader(self, indexes, batch_size, data):
        """
        Gets reference to the data loader which provides batches of <batch_size> sequentially 
        from indexes set. We use SubsetRandomSampler as sampler in returned DataLoader.

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(indexes, np.ndarray), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(batch_size, int), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(type(batch_size))
        
        subsetSampler = IndexedSequentialSampler(indexes)
        # if self.dataset == "IMAGENET":
        #     loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler,pin_memory=True)
        # else:
        loader = DataLoader(dataset=data, batch_size=batch_size, sampler=subsetSampler)
        return loader


    def getTestLoader(self, data, test_batch_size, seed_id=0):
        """
        Implements a random subset sampler for sampling the data from test set.
        
        INPUT:
        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.
        
        test_batch_size: int, Denotes the size of test batch

        seed_id: int, Helps in reporoducing results of random operations
        
        OUTPUT:
        (On Success) Returns the testLoader
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        if self.dataset in ["MNIST","CIFAR10","CIFAR100"]:
            n_datapts = len(data)
            idx = [i for i in range(n_datapts)]
            #np.random.shuffle(idx)

            test_sampler = SubsetRandomSampler(idx)

            testLoader = DataLoader(data, batch_size=test_batch_size, sampler=test_sampler)
            return testLoader

        else:
            raise NotImplementedError


    def loadPartitions(self, lSetPath, uSetPath, valSetPath):

        assert isinstance(lSetPath, str), "Expected lSetPath to be a string."
        assert isinstance(uSetPath, str), "Expected uSetPath to be a string."
        assert isinstance(valSetPath, str), "Expected lSetPath to be a string."

        lSet = np.load(lSetPath, allow_pickle=True)
        uSet = np.load(uSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        #Checking no overlap
        assert len(set(valSet) & set(uSet)) == 0,"Intersection is not allowed between validationset and uset"
        assert len(set(valSet) & set(lSet)) == 0,"Intersection is not allowed between validationset and lSet"
        assert len(set(uSet) & set(lSet)) == 0,"Intersection is not allowed between uSet and lSet"

        return lSet, uSet, valSet


    def saveSets(self, lSet, uSet, activeSet, save_dir):

        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(activeSet, dtype=np.ndarray)

        np.save(f'{save_dir}/lSet.npy', lSet)
        np.save(f'{save_dir}/uSet.npy', uSet)
        np.save(f'{save_dir}/activeSet.npy', activeSet)

        # return f'{save_dir}/lSet.npy', f'{save_dir}/uSet.npy', f'{save_dir}/activeSet.npy'


    def getClassWeightsFromDataset(self, dataset, index_set, bs):
        temp_loader = self.getIndexesDataLoader(indexes=index_set, batch_size=bs, data=dataset)
        return self.getClassWeights(temp_loader)


    def getClassWeights(self, dataloader):

        """
        INPUT
        dataloader: dataLoader
        
        OUTPUT
        Returns a tensor of size C where each element at index i represents the weight for class i. 
        """

        all_labels = []
        for _,y in dataloader:
            all_labels.append(y)
        print("===Computing Imbalanced Weights===")
        
        
        all_labels = np.concatenate(all_labels, axis=0)
        print(f"all_labels.shape: {all_labels.shape}")
        classes = np.unique(all_labels)
        print(f"classes: {classes.shape}")
        num_classes = len(classes)
        freq_count = np.zeros(num_classes, dtype=int)
        for i in classes:
            freq_count[i] = (all_labels==i).sum()
        
        #Normalize
        freq_count = (1.0*freq_count)/np.sum(freq_count)
        print(f"=== Sum(freq_count): {np.sum(freq_count)} ===")
        class_weights = 1./freq_count
        
        class_weights = torch.Tensor(class_weights)
        return class_weights