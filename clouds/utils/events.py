import os
import wandb
from detectron2.utils import comm
from detectron2.utils.events import EventWriter, get_event_storage


def setup_wandb(cfg, args):
    if comm.is_main_process():
        init_args = {
            k.lower(): v
            for k, v in cfg.WANDB.items()
            if isinstance(k, str) and k not in ["config", "name"]
        }
        if "config_exclude_keys" in init_args:
            init_args["config"] = cfg
            init_args["config"]["cfg_file"] = args.config_file
        else:
            init_args["config"] = {
                "output_dir": cfg.OUTPUT_DIR,
                "train": extract_dataset_from_string(cfg.DATASETS.TRAIN),
                "test": extract_dataset_from_string(cfg.DATASETS.TEST),
                "iter": cfg.SOLVER.MAX_ITER,
                "lr": cfg.SOLVER.BASE_LR,
                "batch_size": cfg.SOLVER.IMS_PER_BATCH,
                "cfg_file": args.config_file,
            }

        init_args["group"] = get_base_name(cfg)
        if cfg.WANDB.NAME is not None:
            init_args["name"] = cfg.WANDB.NAME
        else:
            init_args["name"] = get_full_name_xp(init_args["group"], cfg)
        if "debug" in cfg.OUTPUT_DIR:
            init_args["project"] = "debug"
        wandb.init(**init_args)


def get_base_name(cfg):
    source = extract_dataset_from_string(cfg.DATASETS.TRAIN)
    target = extract_dataset_from_string(cfg.DATASETS.TEST)
    return f"{source}_to_{target}"


def get_full_name_xp(base_name, cfg):
    """
    This function will return the base_name concatenated with the batch size and number of epochs which
    is computed using the solver configuration.
    """
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    num_iter = cfg.SOLVER.MAX_ITER
    return f"{base_name}_bs_{batch_size}_iter_{num_iter}"


def extract_dataset_from_string(dataset_str):
    if "gta" in dataset_str[0]:
        return "gta"
    elif "cityscapes" in dataset_str[0]:
        return "city"
    elif "synthia" in dataset_str[0]:
        return "syn"
    elif "bdd" in dataset_str[0]:
        return "bdd"
    elif "mapillary" in dataset_str[0]:
        return "mapillary"
    elif "sd" in dataset_str[0]:
        return "sd"
    elif "acdc" in dataset_str[0]:
        return "acdc"
    else:
        raise NotImplementedError(
            "dataset not supported, add it in extract_dataset_from_string function"
        )


def write_to_wandb(dic):
    if comm.is_main_process():
        wandb.log(dic)


class BaseRule(object):
    def __call__(self, target):
        return target


class IsIn(BaseRule):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(self, target):
        return self.keyword in target


class IsInList(BaseRule):
    def __init__(self, keywords: list):
        self.keyword = keywords

    def __call__(self, target):
        return target in self.keyword


class Prefix(BaseRule):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def __call__(self, target):
        return "/".join([self.keyword, target])


class WandbWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self):
        """
        Args:
            log_dir (str): the directory to save the output events
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._last_write = -1
        self._group_rules = [
            (IsIn("/"), BaseRule()),
            (IsIn("loss"), Prefix("train")),
            # (IsIn("sem_seg"), Prefix("val")),
            (
                IsInList(["lr", "time", "eta_seconds", "rank_data_time", "data_time"]),
                Prefix("stats"),
            ),
        ]

    def write(self):
        storage = get_event_storage()

        def _group_name(scalar_name):
            for rule, op in self._group_rules:
                if rule(scalar_name):
                    return op(scalar_name)
            return scalar_name

        stats = {
            _group_name(name): scalars[0]
            for name, scalars in storage.latest().items()
            if scalars[1] > self._last_write
        }
        if len(stats) > 0:
            self._last_write = max([v[1] for k, v in storage.latest().items()])

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            stats["image"] = [
                wandb.Image(img, caption=img_name)
                for img_name, img, step_num in storage._vis_data
            ]
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:

            def create_bar(tag, bucket_limits, bucket_counts, **kwargs):
                data = [
                    [label, val] for (label, val) in zip(bucket_limits, bucket_counts)
                ]
                table = wandb.Table(data=data, columns=["label", "value"])
                return wandb.plot.bar(table, "label", "value", title=tag)

            stats["hist"] = [create_bar(**params) for params in storage._histograms]

            storage.clear_histograms()

        if len(stats) == 0:
            return
        wandb.log(stats, step=storage.iter)

    def close(self):
        wandb.finish()
