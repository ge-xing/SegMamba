import numpy as np 
from typing import Union, Tuple
import time 

class DataLoaderMultiProcess:
    def __init__(self, dataset, 
                 patch_size,
                 batch_size=2,
                 oversample_foreground_percent=0.33,
                 probabilistic_oversampling=False,
                 print_time=False):
        pass
        self.dataset = dataset
        self.patch_size = patch_size
        # self.annotated_classes_key = annotated_classes_key ## (1, 2, 3 ..)
        self.batch_size = batch_size
        self.keys = [i for i in range(len(dataset))]
        self.thread_id = 0
        self.oversample_foreground_percent = oversample_foreground_percent
        self.need_to_pad = (np.array([0, 0, 0])).astype(int)

        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling
        self.data_shape = None 
        self.seg_shape = None
        self.print_time = print_time

    def determine_shapes(self):
        # load one case
        item = self.dataset.__getitem__(0)
        data, seg, properties = item["data"], item["seg"], item["properties"]
        num_color_channels = data.shape[0]
        num_output_channels = seg.shape[0]
        patch_size = self.patch_size
        data_shape = (self.batch_size, num_color_channels, patch_size[0], patch_size[1], patch_size[2])
        seg_shape = (self.batch_size, num_output_channels, patch_size[0], patch_size[1], patch_size[2])
        return data_shape, seg_shape
    
    def generate_train_batch(self):
        
        selected_keys = np.random.choice(self.keys, self.batch_size, True, None)
        if self.data_shape is None:
            self.data_shape, self.seg_shape = self.determine_shapes()

        data_all = np.zeros(self.data_shape, dtype=np.float32)
        data_all_global = np.zeros(self.data_shape, dtype=np.float32)
        seg_all_global = np.zeros(self.seg_shape, dtype=np.float32)
        data_global = None
        seg_global = None
        seg_all = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []

        index = 0
        for j, key in enumerate(selected_keys):

            force_fg = self.get_do_oversample(j)
            s = time.time()
            item = self.dataset.__getitem__(key)
            e = time.time()
            if self.print_time:
                print(f"read single data time is {e - s}")
            # print(f"read data time is {e - s}")
            data, seg, properties = item["data"], item["seg"], item["properties"]
            
            if "data_global" in item:
                data_global = item["data_global"]
            
            if "seg_global" in item:
                seg_global = item["seg_global"]

            case_properties.append(properties)
            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            
            s = time.time()
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            e = time.time()
            if self.print_time:
                print(f"get bbox time is {e - s}")
            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]


            s = time.time()
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            # print(f"box is {bbox_lbs, bbox_ubs}, padding is {padding}")
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)

            if data_global is not None :
                data_all_global[j] = data_global

            if seg_global is not None :
                seg_all_global[j] = seg_global


            e = time.time()
            if self.print_time:
                print(f"box is {bbox_lbs, bbox_ubs}, padding is {padding}")
                print(f"setting data value time is {e - s}")
                
        
        if data_global is None:
            return {'data': data_all,
                    'seg': seg_all, 'properties': case_properties, 
                    'keys': selected_keys}
    
        return {'data': data_all, "data_global": data_all_global,
                    "seg_global": seg_all_global, 
                    'seg': seg_all, 'properties': case_properties, 
                    'keys': selected_keys}

    def __next__(self):
    
        return self.generate_train_batch() 
    
    def set_thread_id(self, thread_id):
        self.thread_id = thread_id
    
    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent
    
    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
            if overwrite_class is not None:
                assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                    'have class_locations (missing key)'
            # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
            # class_locations keys can also be tuple
            eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

            # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
            # strange formulation needed to circumvent
            # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            # tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
            # if any(tmp):
            #     if len(eligible_classes_or_regions) > 1:
            #         eligible_classes_or_regions.pop(np.where(tmp)[0][0])

            if len(eligible_classes_or_regions) == 0:
                # this only happens if some image does not contain foreground voxels at all
                selected_class = None
                if verbose:
                    print('case does not contain any foreground classes')
            else:
                # I hate myself. Future me aint gonna be happy to read this
                # 2022_11_25: had to read it today. Wasn't too bad
                selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                    (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
            # print(f'I want to have foreground, selected class: {selected_class}')
          
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs