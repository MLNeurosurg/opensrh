"""PyTorch datasets designed to work with OpenSRH.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import os
import json
import logging
from collections import Counter
from typing import Optional, List, Union, TypedDict
import random

from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import Compose

from opensrh.datasets.improc import process_read_im, get_srh_base_aug

get_chnl_min = lambda im: im.min(dim=1).values.min(dim=1).values.squeeze()
get_chnl_max = lambda im: im.max(dim=1).values.max(dim=1).values.squeeze()


class PatchData(TypedDict):
    image: Optional[torch.Tensor]
    label: Optional[torch.Tensor]
    path: Optional[List[str]]


class PatchContrastiveData(TypedDict):
    image: Optional[List[torch.Tensor]]
    label: Optional[torch.Tensor]
    path: Optional[List[str]]


class SRHBaseDataset(Dataset):
    """Abstract base class for OpenSRH dataset.

    Args:
        data_root: root OpenSRH directory
        studies: either a string in {"train", "val"} for the default train/val
            dataset split, or a list of strings representing patient IDs
        transform: a callable object for image transformation
        target_transform: a callable object for label transformation
        check_images_exist: a flag representing whether to check every image
            file exists in data_root. Turn this on for debugging, turn it off
            for speed.
    """

    def __init__(self, data_root: str, studies: Union[str, List[str]],
                 transform: callable, target_transform: callable,
                 check_images_exist: bool) -> None:
        """Inits the base abstract dataset

        Populate each attribute and walk through each slide to look for patches
        """

        self.data_root_ = data_root
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.check_images_exist_ = check_images_exist
        self.get_all_meta()
        self.get_study_list(studies)

        # Walk through each study
        self.instances_ = []
        for p in tqdm(self.studies_):
            self.instances_.extend(self.get_study_instances(p))

    def get_all_meta(self):
        """Read in all metadata files."""
        try:
            with open(os.path.join(self.data_root_,
                                   "meta/opensrh.json")) as fd:
                self.metadata_ = json.load(fd)
        except Exception as e:
            logging.critical("Failed to locate dataset.")
            raise e

        logging.info(f"Locate OpenSRH dataset at {self.data_root_}")
        return

    def get_study_list(self, studies):
        """Get a list of studies from default split or list of IDs."""
        if isinstance(studies, str):
            try:
                with open(
                        os.path.join(self.data_root_,
                                     "meta/train_val_split.json")) as fd:
                    train_val_split = json.load(fd)
            except Exception as e:
                logging.critical("Failed to locate preset train/val split.")
                raise e

            if studies == "train":
                self.studies_ = train_val_split["train"]
            elif studies in ["valid", "val"]:
                self.studies_ = train_val_split["val"]
            else:
                return ValueError(
                    "studies split must be one of [\"train\", \"val\"]")
        elif isinstance(studies, List):
            self.studies_ = studies
        else:
            raise ValueError("studies must be a string representing " +
                             "train/val split or a list of study numbers")
        return

    def get_study_instances(self, patient: str):
        """Get all instances from one study."""
        slide_instances = []
        logging.debug(patient)
        if self.check_images_exist_:
            tiff_file_exist = lambda im_p: (os.path.exists(im_p) and
                                            is_image_file(im_p))
        else:
            tiff_file_exist = lambda _: True

        def check_add_patches(patches: List[str]):
            for p in patches:
                im_p = os.path.join(self.data_root_, p)
                if tiff_file_exist(im_p):
                    slide_instances.append(
                        (im_p, self.metadata_[patient]["class"]))
                else:
                    logging.warning(f"Bad patch: unable to locate {im_p}")

        for s in self.metadata_[patient]["slides"]:
            if self.metadata_[patient]["class"] == "normal":
                check_add_patches(
                    self.metadata_[patient]["slides"][s]["normal_patches"])
            else:
                check_add_patches(
                    self.metadata_[patient]["slides"][s]["tumor_patches"])
        logging.debug(f"patient {patient} patches {len(slide_instances)}")
        return slide_instances

    def process_classes(self):
        """Look for all the labels in the dataset.

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [i[1] for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))
        return

    def get_weights(self):
        """Count number of instances for each class, and computes weights."""
        # Get classes
        self.process_classes()
        all_labels = [self.class_to_idx_[i[1]] for i in self.instances_]

        # Count number of slides in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: {}".format(self.weights_))
        return self.weights_

    def replicate_balance_instances(self):
        """resample the instances list to balance each class."""
        all_labels = [i[1] for i in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_instances_ = []
        for l in sorted(set(all_labels)):
            instances_l = [i for i in self.instances_ if i[1] == l]
            random.shuffle(instances_l)
            instances_l = instances_l * (val_sample // len(instances_l) + 1)
            all_instances_.extend(sorted(instances_l[:val_sample]))

        self.instances_ = all_instances_
        return

    def __len__(self):
        return len(self.instances_)


class SRHClassificationDataset(SRHBaseDataset):
    """OpenSRH Patch dataset for classification (cross entropy)

    Additional args:
        balance_patch_per_class: whether to over sample patches so every class
            have the same number of patches
    """

    def __init__(self,
                 data_root: str,
                 studies: Union[str, List[str]],
                 transform: callable = Compose(get_srh_base_aug()),
                 target_transform: callable = torch.tensor,
                 balance_patch_per_class: bool = False,
                 check_images_exist: bool = False) -> None:

        super().__init__(data_root=data_root,
                         studies=studies,
                         transform=transform,
                         target_transform=target_transform,
                         check_images_exist=check_images_exist)

        if balance_patch_per_class:
            self.replicate_balance_instances()
        self.get_weights()

    def __getitem__(self, idx: int) -> PatchData:
        """Retrieve a patch specified by idx"""
        imp, target = self.instances_[idx]
        target = self.class_to_idx_[target]

        # Read image
        try:
            im: torch.Tensor = process_read_im(imp)
        except:
            logging.warning("bad_file - {}".format(imp))
            return {"image": None, "label": None, "path": [None]}

        logging.debug(f"before xform patch shape {im.shape}")
        logging.debug(f"before xform channel mean  {im.mean(dim=[1,2])}")
        logging.debug(f"before xform channel min   {get_chnl_min(im)}")
        logging.debug(f"before xform channel max   {get_chnl_max(im)}")

        # Perform transformations
        if self.transform_ is not None:
            im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        logging.debug(f"after  xform patch shape {im.shape}")
        logging.debug(f"after  xform channel mean  {im.mean(dim=[1,2])}")
        logging.debug(f"after  xform channel min   {get_chnl_min(im)}")
        logging.debug(f"after  xform channel max   {get_chnl_max(im)}")

        return {"image": im, "label": target, "path": [imp]}


class SRHContrastiveDataset(SRHBaseDataset):
    """Patch Contrastive Dataset. Performs multiple transformations on a patch

    Can be used for SimCLR and SupCon experiments

    Additional args:
        num_transforms: number of transformation on each patch sampled
        balance_patch_per_class: whether to over sample patches so every class
            have the same number of patches
    """

    def __init__(self,
                 data_root: str,
                 studies: Union[str, List[str]],
                 num_transforms: int = 2,
                 transform: callable = Compose(get_srh_base_aug()),
                 target_transform: callable = torch.tensor,
                 balance_patch_per_class: bool = False,
                 check_images_exist: bool = False) -> None:

        super().__init__(data_root=data_root,
                         studies=studies,
                         transform=transform,
                         target_transform=target_transform,
                         check_images_exist=check_images_exist)
        self.num_transforms_ = num_transforms
        if balance_patch_per_class:
            self.replicate_balance_instances()
        self.get_weights()

    def __getitem__(self, idx: int) -> PatchContrastiveData:
        imp, target = self.instances_[idx]
        target = self.class_to_idx_[target]

        # Read image
        try:
            im: torch.Tensor = process_read_im(imp)
        except:
            logging.warning("bad_file - {}".format(imp))
            return {"image": None, "label": None, "path": [None]}

        # Perform transformations
        if self.transform_ is not None:
            im = [self.transform_(im) for _ in range(self.num_transforms_)]
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [imp]}
