# Uni-NLX: Unifying Textual Explanations for Vision and Vision-Language Tasks

<br>
<br>
<p align="center">
<img src="demo_uninlx.png" width="784"/>
  </p>

### Requirements
- [PyTorch](https://pytorch.org/) 1.8 or higher
- [CLIP](https://github.com/openai/CLIP) (install with `pip install git+https://github.com/openai/CLIP.git`)
- [transformers](https://huggingface.co/docs/transformers/index) (install with `pip install transformers`)
- [cococaption](https://github.com/ruotianluo/coco-caption/tree/ea20010419a955fed9882f9dcc53f2dc1ac65092) 

### Images Download
- [COCO](https://cocodataset.org/#download) `train2014` and `val2014` images<br>
- [MPI](http://human-pose.mpi-inf.mpg.de/#download). Rename to `mpi` <br>
- [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/). Rename to `flickr30k` <br>
- [VCR](https://visualcommonsense.com/download/). Rename to `vcr` <br>
- [ImageNet (ILSVRC2012)](https://www.image-net.org/download.php)
- [Visual Genome v1.2](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)

### Code
`train_nlx.py`: script for training only<br>
`test_datasets.py`: script for validation/testing for all epochs on all 7 NLE tasks<br>
`clip_model.py`: script for vision backbone we use (CLIP visual encoder)<br>

### Models

### Results
