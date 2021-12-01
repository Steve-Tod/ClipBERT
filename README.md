# ClipBERT

[Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling](https://arxiv.org/abs/2102.06183) 

[CVPR 2021](http://cvpr2021.thecvf.com/), Oral, [Best Student Paper Honorable Mention](http://cvpr2021.thecvf.com/node/329).

[Jie Lei](http://www.cs.unc.edu/~jielei/)\*, [Linjie Li](https://www.linkedin.com/in/linjie-li/)\*,
[Luowei Zhou](https://luoweizhou.github.io/), [Zhe Gan](https://zhegan27.github.io/), 
[Tamara L. Berg](http://tamaraberg.com/), [Mohit Bansal](http://www.cs.unc.edu/~mbansal/),
[Jingjing Liu](https://www.linkedin.com/in/jingjing-liu-65703431/)

Official PyTorch code for ClipBERT, an efficient framework for 
end-to-end learning for image-text and video-text tasks. 
It takes raw videos/images + text as inputs, and outputs task predictions.
ClipBERT is designed based on 2D CNNs and transformers, and uses a sparse sampling strategy 
to enable efficient end-to-end video-and-language learning. In this repository, 
we support end-to-end pretraining and finetuning for the following tasks:

- Image-text pretraining on COCO and VG captions.
- Text-to-video retrieval finetuning on MSRVTT, DiDeMo, and ActivityNet Captions.
- Video-QA finetuning on TGIF-QA and MSRVTT-QA.
- Image-QA finetuning on VQA 2.0.

It is also feasible and easy to add other image-text or video-text tasks for pretraining and finetuning. 


## Requirements 
We provide a Docker image for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.
We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.


## Getting Started

### General

1. Create a folder that stores pretrained models, all the data, and results.
    ```bash
    PATH_TO_STORAGE=/path/to/your/data/
    mkdir -p $PATH_TO_STORAGE/txt_db  # annotations
    mkdir -p $PATH_TO_STORAGE/vis_db  # image and video 
    mkdir -p $PATH_TO_STORAGE/finetune  # finetuning results
    mkdir -p $PATH_TO_STORAGE/pretrained  # pretrained models
    ```

2. Download pretrained models.

    Our e2e pretrained ClipBERT model (849MB), can be downloaded with the following command.
    ```bash
    bash scripts/download_pretrained.sh $PATH_TO_STORAGE
    ```
    This pretrained model can be used for finetuning on video-text tasks and image-text tasks.
    For your convenience, this script will also download `bert-base-uncased` and `grid-feat-vqa` 
    model weights, which are used as initialization for pretraining.  

### Downstream Task Finetuning

#### Image Question Answering (VQA)
1. Download data
    ```bash
    # outside the container
    # download COCO and VG data
    bash scripts/download_coco_vg.sh $PATH_TO_STORAGE
    # download VQA annotations
    bash scripts/download_vqa.sh $PATH_TO_STORAGE
    ```

2. Finetuning
    ```bash
    # inside the container
    PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=4 src/tasks/run_vqa.py \
        --config src/configs/vqa_base_resnet50.json \
        --output_dir $OUTPUT_DIR
    ```

3. Inference
    ```bash
    # inside the container
    PYTHONPATH=. python python -m torch.distributed.launch --nproc_per_node=4 src/tasks/run_vqa.py \
      --do_inference 1 --output_dir $OUTPUT_DIR \
      --inference_split val --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 64
    ```


## Pretraining
1. Download data
    ```bash
    # outside the container
    bash scripts/download_coco_vg.sh $PATH_TO_STORAGE
    ```

2. Pretraining
    ```bash
    #inside the container
    horovodrun -np 8 python src/pretrain/run_pretrain.py \
        --config src/configs/pretrain_image_text_base_resnet50_mlm_itm.json \
        --output_dir $OUTPUT_DIR 
    ``` 

## Data Preprocessing
ClipBERT takes raw video and text as inputs, there is no need to do feature extraction. 
However, to improve data loading speed, we use LMDB to store the raw image and video files. 
You can use the following script to convert a list of videos with file extensions `mp4` and `avi` into LMDB:
    
```bash
# outside the container
python src/preprocessing/file2lmdb.py \
    --data_root /path/to/videos \
    --lmdb_save_dir /path/to/save/lmdb \
    --ext avi mp4 \
    --file_type video 
```

For images, use appropriate file extensions for `--ext` and `--file_type image`. 
Text annotation files are reorganized into `jsonl` files, 
see example preprocessed files downloaded by the scripts in [scripts/](scripts).   


## Citation

If you find this code useful for your research, please consider citing:
```
@inproceedings{lei2021less,
  title={Less is More: ClipBERT for Video-and-Language Learningvia Sparse Sampling},
  author={Lei, Jie and Li, Linjie and Zhou, Luowei and Gan, Zhe and Berg, Tamara L. and Bansal, Mohit and Liu, Jingjing},
  booktitle={CVPR},
  year={2021}
}
```

## Acknowledgement
We thank [Yen-Chun Chen](https://scholar.google.com/citations?user=Gptgy4YAAAAJ&hl=en), 
[Ruotian Luo](https://ttic.uchicago.edu/~rluo/), and other members and interns at 
[Microsoft Multimodal AI](https://multimodalai.azurewebsites.net/people/members) 
for their helpful discussions.
We also thank the anonymous reviewers for their constructive feedback.

This code used resources from [transformers](https://github.com/huggingface/transformers), 
[UNITER](https://github.com/ChenRocks/UNITER), [HERO](https://github.com/linjieli222/HERO), 
[grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa), 
[SlowFast](https://github.com/facebookresearch/SlowFast), 
[Detectron2](https://github.com/facebookresearch/detectron2). 
The code is implemented using [PyTorch](https://github.com/pytorch/pytorch), 
with multi-GPU support from [Horovod](https://github.com/horovod/horovod) 
and mixed precision support from [apex](https://github.com/NVIDIA/apex).  We thank the authors for open-sourcing their awesome projects.



## License

MIT

