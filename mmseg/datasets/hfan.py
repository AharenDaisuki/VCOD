import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
from PIL import Image
from functools import reduce
from collections import OrderedDict
from prettytable import PrettyTable

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
# from .builder import DATASETS
# from .custom_VOS import CustomDatasetVOS

# TODO: move customized dataset to basesegdataset
@DATASETS.register_module()
class CustomDatasetVOS(BaseSegDataset):
    """Custom dataset for change detection. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img1_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── img2_dir2
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img1_dir (str): Path to first image directory
        img2_dir (str): Path to second image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img1_dir,
                 img2_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 if_visualize=False,
                 ):
        # self.pipeline = ComposeWithVisualization(pipeline, if_visualize=if_visualize)
        self.img1_dir = img1_dir
        self.img2_dir = img2_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None  # map from old class index to new class index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img1_dir):
                self.img1_dir = osp.join(self.data_root, self.img1_dir)
                self.img2_dir = osp.join(self.data_root, self.img2_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img1_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img1_prefix'] = self.img1_dir
        results['img2_prefix'] = self.img2_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    # def evaluate(self,
    #              results,
    #              metric='mIoU',
    #              logger=None,
    #              efficient_test=False,
    #              **kwargs):
    #     """Evaluate the dataset.

    #     Args:
    #         results (list): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated. 'mIoU',
    #             'mDice' and 'mFscore' are supported.
    #         logger (logging.Logger | None | str): Logger used for printing
    #             related information during evaluation. Default: None.

    #     Returns:
    #         dict[str, float]: Default metrics.
    #     """

    #     eval_results = {}
    #     gt_seg_maps = self.get_gt_seg_maps(efficient_test)
    #     if self.CLASSES is None:
    #         num_classes = len(
    #             reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
    #     else:
    #         num_classes = len(self.CLASSES)
    #     # print(results[0].dtype)
    #     # print(gt_seg_maps[0].dtype)
    #     ret_metrics = vos_eval_metrics(
    #         results,
    #         gt_seg_maps,
    #         num_classes,
    #         self.ignore_index,
    #         metric,
    #         label_map=self.label_map,
    #         reduce_zero_label=self.reduce_zero_label)

    #     # if self.CLASSES is None:
    #     #     class_names = tuple(range(num_classes))
    #     # else:
    #     #     class_names = self.CLASSES

    #     # summary table
    #     # ret_metrics_summary = OrderedDict({
    #     #     ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
    #     #     for ret_metric, ret_metric_value in ret_metrics.items()
    #     # })
    #     VOS_metrics = ['JM', 'JO', 'FM', 'FO']
    #     ret_metrics_VOS = OrderedDict({
    #         key: np.round(np.nanmean(ret_metrics.pop(key)) * 100, 2)
    #         for key in VOS_metrics
    #     })

    #     # # each class table
    #     # ret_metrics.pop('aAcc', None)
    #     # ret_metrics_class = OrderedDict({
    #     #     ret_metric: np.round(ret_metric_value * 100, 2)
    #     #     for ret_metric, ret_metric_value in ret_metrics.items()
    #     # })
    #     # ret_metrics_class.update({'Class': class_names})
    #     # ret_metrics_class.move_to_end('Class', last=False)

    #     # for logger
    #     # class_table_data = PrettyTable()
    #     # for key, val in ret_metrics_class.items():
    #     #     class_table_data.add_column(key, val)

    #     # summary_table_data = PrettyTable()
    #     # for key, val in ret_metrics_summary.items():
    #     #     if key == 'aAcc':
    #     #         summary_table_data.add_column(key, [val])
    #     #     else:
    #     #         summary_table_data.add_column('m' + key, [val])

    #     # VOS_table_data = PrettyTable()
    #     # for key, val in ret_metrics_VOS.items():
    #     #     VOS_table_data.add_column(key, [val])

    #     # print_log('per class results:', logger)
    #     # print_log('\n' + class_table_data.get_string(), logger=logger)
    #     # print_log('Summary:', logger)
    #     # print_log('\n' + summary_table_data.get_string(), logger=logger)
    #     # print_log('\n' + VOS_table_data.get_string(), logger=logger)

    #     # each metric dict
    #     # for key, value in ret_metrics_summary.items():
    #     #     if key == 'aAcc':
    #     #         eval_results[key] = value / 100.0
    #     #     else:
    #     #         eval_results['m' + key] = value / 100.0

    #     # each metric dict
    #     for key, value in ret_metrics_VOS.items():
    #         eval_results[key] = value / 100.0

    #     # ret_metrics_class.pop('Class', None)
    #     # for key, value in ret_metrics_class.items():
    #     #     eval_results.update({
    #     #         key + '.' + str(name): value[idx] / 100.0
    #     #         for idx, name in enumerate(class_names)
    #     #     })

    #     if mmcv.is_list_of(results, str):
    #         for file_name in results:
    #             os.remove(file_name)
    #     return eval_results

# for moca
@DATASETS.register_module()
class MoCAMaskFlowDataset(CustomDatasetVOS):
    """VOS dataset.

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('Background', 'Foreground')

    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None

        result_files = self.results2img(results, imgfile_prefix, to_label_id)
        return result_files, tmp_dir