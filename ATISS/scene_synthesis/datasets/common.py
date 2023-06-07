# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

from collections import Counter

import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset


class InfiniteDataset(IterableDataset):
    """Decorate any Dataset instance to provide an infinite IterableDataset
    version of it."""
    def __init__(self, dataset, shuffle=True):
        super().__init__()
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        N = len(self.dataset)

        # Split the work if we have multiple workers
        # see https://pytorch.org/stable/data.html
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = 0
            end = N
        else:
            num_workers = worker_info.num_workers
            per_worker = (N + num_workers - 1) // num_workers
            start = worker_info.id * per_worker
            end = min(start + per_worker, N)

        indices = np.arange(start, end)
        while True:
            if self.shuffle:
                np.random.shuffle(indices)
            for i in indices:
                yield self.dataset[i]


class BaseDataset(Dataset):
    """Implements the interface for all datasets that consist of scenes."""
    def __init__(self, scenes):
        assert len(scenes) > 0
        self.scenes = scenes

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        return self.scenes[idx]

    @property
    def class_labels(self):
        raise NotImplementedError()

    @property
    def n_classes(self):
        return len(self.class_labels)

    @property
    def object_types(self):
        raise NotImplementedError()

    @property
    def n_object_types(self):
        """The number of distinct objects contained in the scenes."""
        return len(self.object_types)

    @property
    def room_types(self):
        return set([si.scene_type for si in self.scenes])

    @property
    def count_objects_in_rooms(self):
        return Counter([len(si.bboxes) for si in self.scenes])

    def post_process(self, s):
        return s

    @staticmethod
    def with_valid_scene_ids(invalid_scene_ids):
        def inner(scene):
            return scene if scene.scene_id not in invalid_scene_ids else False
        return inner

    @staticmethod
    def with_scene_ids(scene_ids):
        def inner(scene):
            return scene if scene.scene_id in scene_ids else False
        return inner

    @staticmethod
    def with_room(scene_type):
        def inner(scene):
            return scene if scene_type in scene.scene_type else False
        return inner

    @staticmethod
    def room_smaller_than_along_axis(max_size, axis=1):
        def inner(scene):
            return scene if scene.bbox[1][axis] <= max_size else False
        return inner

    @staticmethod
    def room_larger_than_along_axis(min_size, axis=1):
        def inner(scene):
            return scene if scene.bbox[0][axis] >= min_size else False
        return inner

    @staticmethod
    def floor_plan_with_limits(limit_x, limit_y, axis=[0, 2]):
        def inner(scene):
            min_bbox, max_bbox = scene.floor_plan_bbox
            t_x = max_bbox[axis[0]] - min_bbox[axis[0]]
            t_y = max_bbox[axis[1]] - min_bbox[axis[1]]
            if t_x <= limit_x and t_y <= limit_y:
                return scene
            else:
                False
        return inner

    @staticmethod
    def with_valid_boxes(box_types):
        def inner(scene):
            for i in range(len(scene.bboxes)-1, -1, -1):
                if scene.bboxes[i].label not in box_types:
                    scene.bboxes.pop(i)
            return scene
        return inner

    @staticmethod
    def without_box_types(box_types):
        def inner(scene):
            for i in range(len(scene.bboxes)-1, -1, -1):
                if scene.bboxes[i].label in box_types:
                    scene.bboxes.pop(i)
            return scene
        return inner

    @staticmethod
    def with_generic_classes(box_types_map):
        def inner(scene):
            for box in scene.bboxes:
                # Update the box label based on the box_types_map
                box.label = box_types_map[box.label]
            return scene
        return inner

    @staticmethod
    def with_valid_bbox_jids(invalid_bbox_jds):
        def inner(scene):
            return (
                False if any(b.model_jid in invalid_bbox_jds for b in scene.bboxes)
                else scene
            )
        return inner

    @staticmethod
    def at_most_boxes(n):
        def inner(scene):
            return scene if len(scene.bboxes) <= n else False
        return inner

    @staticmethod
    def at_least_boxes(n):
        def inner(scene):
            return scene if len(scene.bboxes) >= n else False
        return inner

    @staticmethod
    def with_object_types(objects):
        def inner(scene):
            return (
                scene if all(b.label in objects for b in scene.bboxes)
                else False
            )
        return inner

    @staticmethod
    def contains_object_types(objects):
        def inner(scene):
            return (
                scene if any(b.label in objects for b in scene.bboxes)
                else False
            )
        return inner

    @staticmethod
    def without_object_types(objects):
        def inner(scene):
            return (
                False if any(b.label in objects for b in scene.bboxes)
                else scene
            )
        return inner

    @staticmethod
    def filter_compose(*filters):
        def inner(scene):
            s = scene
            fs = iter(filters)
            try:
                while s:
                    s = next(fs)(s)
            except StopIteration:
                pass
            return s
        return inner


class BaseScene(object):
    """Contains all the information for a scene."""
    def __init__(self, scene_id, scene_type, bboxes):
        self.bboxes = bboxes
        self.scene_id = scene_id
        self.scene_type = scene_type

    def __str__(self):
        return "Scene: {} of type: {} contains {} bboxes".format(
            self.scene_id, self.scene_type, self.nobjects
        )

    @property
    def nobjects(self):
        """Number of bounding boxes / objects in a Scene."""
        return len(self.bboxes)

    @property
    def object_types(self):
        """The set of object types in this scene."""
        return sorted(set(b.label for b in self.bboxes))

    @property
    def n_object_types(self):
        """Number of distinct objects in a Scene."""
        return len(self.object_types)

    def ordered_bboxes_with_centroid(self):
        centroids = np.array([b.centroid for b in self.bboxes])
        ordering = np.lexsort(centroids.T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_labels(self, all_labels):
        centroids = np.array([b.centroid for b in self.bboxes])
        int_labels = np.array([[b.int_label(all_labels)] for b in self.bboxes])
        ordering = np.lexsort(np.hstack([centroids, int_labels]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes
