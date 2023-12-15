# Prepare Datasets for CLOUDS

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

CLOUDS has builtin support for a few datasets, the datasets are assumed to exist in `datasets/` directory.
Under this directory, detectron2 will look for datasets in the structure described below.

## Setup Datasets

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `datasets/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `datasets/gta5`.

**Synthia:** Please, download SYNTHIA-RAND-CITYSCAPES from
[here](http://synthia-dataset.net/downloads/) and extract it to `datasets/synthia`.

**ACDC:** Please, download rgb_anon_trainvaltest.zip and
gt_trainval.zip from [here](https://acdc.vision.ee.ethz.ch/download) and
extract them to `datasets/acdc`.

**BDD100K:** Please, download the `10K Images` and `Segmentation` from
[here](https://bdd-data.berkeley.edu/portal.html#download) and extract it
to `datasets/bdd100k`.

**Mapillary:** Please, download the mapillary-vistas-dataset_public_v1.2.zip
from [here](https://www.mapillary.com/dataset/vistas) and extract it
to `datasets/mapillary_vistas`.

The final folder structure should look like this:


```none
HRDA
├── ...
├── datasets
│   ├── acdc_list
│   ├── acdc
│   ├── bdd_list
│   ├── bdd100k
│   ├── cityscapes_list
│   ├── cityscapes
│   ├── gta_list
│   ├── gta
│   ├── mapillary_list
│   ├── mapillary_vistas
│   ├── synthia_list
│   ├── synthia
├── ...
```

**Data Preprocessing:** Finally, please run the following scripts to convert the label IDs to the
train IDs:

```shell
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
python tools/convert_datasets/synthia.py data/synthia/ --nproc 8
python tools/convert_datasets/mapillary.py data/mapillary/ --nproc 8
```

