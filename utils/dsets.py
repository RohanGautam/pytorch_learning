from collections import namedtuple
from functools import lru_cache
import functools
import glob
import csv
import os
import re
import SimpleITK as sitk
import numpy as np
import copy
import torch
from utils.conversions import XyzTuple, xyz2irc
from utils.custom_caching import getCache

from torch.utils.data import Dataset

raw_cache = getCache('part2ch10_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz'
)

# The decorator is to cache the last 1 result of the function.
# If the operation is expensive then this is helpful, as it will not redo the operation a second time around.


@lru_cache(maxsize=1)
def getCandidateInfo():
    # `mhd` files contain metadata. `glob` is used for pattern matching files
    mhd_list = glob.glob('data_unversioned/luna/subset*/*.mhd')

    def no_ext(
        x): return re.search(r'(.*)\.mhd', x).group(1)

    seies_names_on_disk = {no_ext(os.path.split(p)[-1]) for p in mhd_list}

    diameter_dict = {}
    with open('data_unversioned/luna/annotations.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotations_center = tuple(map(float, row[1:4]))
            annotation_diameter = float(row[4])
            # series_uid represents one "cross section", and can have multiple annotations
            diameter_dict.setdefault(series_uid, []).append(
                (annotations_center, annotation_diameter)
            )
    candiate_info_list = []
    with open('data_unversioned/luna/candidates.csv', 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            # doesnt makes sense to care about info on subsets we don't have on disk
            if series_uid not in seies_names_on_disk:
                continue

            candidates_center = tuple(map(float, row[1:4]))
            is_nodule = bool(int(row[4]))
            candidate_diameter = 0.0
            for annotation_tuple in diameter_dict.get(series_uid, []):
                center_xyz, diameter = annotation_tuple
                # checking if the centers are close by, and correlating the diameter if so
                deltas = [abs(candidates_center[i] - center_xyz[i])
                          for i in range(3)]
                close_enough = all([i < diameter/4 for i in deltas])
                # if it's *not* close, then we treat it as a nodule of zero diameter
                if close_enough:
                    candidate_diameter = diameter
            candiate_info_list.append(CandidateInfoTuple(
                is_nodule, candidate_diameter, series_uid, candidates_center
            ))
    candiate_info_list.sort(reverse=True)

    # we're doing this so it'll be easier to get a representative partition during
    # our train test split
    return candiate_info_list


class Ct:
    def __init__(self, series_uid) -> None:
        mhd_path = glob.glob(
            f'data_unversioned/luna/subset*/{series_uid}.mhd')[0]
        # implicitly consumes associated raw file
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        # Hounsfield Units, a measure of "density"
        self.hu_a = ct_arr
        self.series_uid = series_uid
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr(
                [self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(self, val_stride=0, is_validation_bool=None, series_uid=None):
        # copy so as to not affect the cache
        self.candidate_info_list = copy.copy(getCandidateInfo())
        # an option to examine only one series uid
        if series_uid:
            self.candidate_info_list = [
                x for x in self.candidate_info_list if x.series_uid == series_uid]

        # if true, will return a validation set
        if is_validation_bool:
            assert val_stride and val_stride > 0
            self.candidate_info_list = self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
        elif val_stride > 0:
            # delete the images at the validation stride locations.
            # used to clear/ return trianing images where that isn't needed
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidate_info_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )
