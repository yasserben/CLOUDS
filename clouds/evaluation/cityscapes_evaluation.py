# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import logging
import numpy as np
import os
import tempfile
from collections import OrderedDict
import torch
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator
from my_utils import *


class CityscapesEvaluator(DatasetEvaluator):
    """
    Base class for evaluation using cityscapes API.
    """

    def __init__(self, dataset_name, save_pl=False, output_dir=None):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._save_pl = save_pl
        self._output_folder = output_dir

    def reset(self):
        if self._save_pl:
            create_directory(self._output_folder)
            self._working_dir = os.path.join(self._output_folder, "cityscapes_eval_pl")
            create_directory(self._working_dir)
            self._temp_dir = self._working_dir
            # self._working_dir = tempfile.TemporaryDirectory(dir=self._output_folder, prefix="cityscapes_", suffix="pl")
        else:
            self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
            self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        assert (
            comm.get_local_size() == comm.get_world_size()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if not self._save_pl:
            if self._temp_dir != self._working_dir.name:
                self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(
                self._temp_dir
            )
        )


class CityscapesSemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import trainId2label
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy()
            pred = 255 * np.ones(output.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)


    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(
            os.path.join(gt_dir, "*", "*_gtFine_labelIds.png")
        )
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(
                cityscapes_eval.getPrediction(cityscapes_eval.args, gt)
            )
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        ret["sem_seg"] = {
            "mIoU": 100.0 * results["averageScoreClasses"],
            "IoU.road": 100.0 * results["classScores"]["road"],
            "IoU.sidewalk": 100.0 * results["classScores"]["sidewalk"],
            "IoU.building": 100.0 * results["classScores"]["building"],
            "IoU.wall": 100.0 * results["classScores"]["wall"],
            "IoU.fence": 100.0 * results["classScores"]["fence"],
            "IoU.pole": 100.0 * results["classScores"]["pole"],
            "IoU.traffic light": 100.0 * results["classScores"]["traffic light"],
            "IoU.traffic sign": 100.0 * results["classScores"]["traffic sign"],
            "IoU.vegetation": 100.0 * results["classScores"]["vegetation"],
            "IoU.terrain": 100.0 * results["classScores"]["terrain"],
            "IoU.sky": 100.0 * results["classScores"]["sky"],
            "IoU.person": 100.0 * results["classScores"]["person"],
            "IoU.rider": 100.0 * results["classScores"]["rider"],
            "IoU.car": 100.0 * results["classScores"]["car"],
            "IoU.truck": 100.0 * results["classScores"]["truck"],
            "IoU.bus": 100.0 * results["classScores"]["bus"],
            "IoU.train": 100.0 * results["classScores"]["train"],
            "IoU.motorcycle": 100.0 * results["classScores"]["motorcycle"],
            "IoU.bicycle": 100.0 * results["classScores"]["bicycle"],
        }
        if not self._save_pl:
            self._working_dir.cleanup()
        return ret
