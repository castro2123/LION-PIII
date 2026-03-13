# 🦁 LION Multimodal Vision Intelligence

<img src="./assets/LION_logo.png" width="120">

**Interactive Multimodal Vision System based on LION**
Empowering image and video understanding with spatial reasoning, semantic graphs and symbolic logic.

---

# Overview

This project extends the **LION (Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge)** framework and provides an **interactive application for multimodal visual understanding**.

The system integrates **deep learning, scene graphs and symbolic reasoning** to analyze both **images and videos**.

The application allows users to:

* Detect objects
* Generate captions
* Ask questions about images or videos
* Extract spatial relationships
* Build semantic scene graphs
* Cluster visual objects
* Perform symbolic reasoning using **Prolog**

All functionalities are available through an **interactive Streamlit interface**.

---

# Key Features

## Image Analysis

The system supports multiple analysis modes for images:

### Caption Generation

Generate natural language descriptions using LION.

### Object Detection

Detect objects using **YOLO + LION semantic tagging**.

### Object Clustering

Group detected objects based on visual similarity.

### Interactive Question Answering

Ask questions about the image content.

Example:

```
"What are the people doing?"
```

### Spatial Scene Graph

Extract **spatial relationships between objects**, such as:

* left of
* right of
* above
* below
* near

### Semantic Scene Graph

Generate high-level semantic relations extracted from captions.

Example:

```
(person, riding, horse)
(dog, sitting on, grass)
```

### Prolog Symbolic Representation

Convert scene descriptions into **Prolog facts and rules**, enabling logical reasoning over visual scenes.

Example:

```
located(person, park).
near(dog, person).
talking(person1, person2).
```

---

# Video Analysis

The system also supports **video processing**.

### Video Captioning

Generate captions for key frames in the video.

### Video Object Detection

Detect objects frame by frame using YOLO.

### Video Clustering

Cluster objects detected across frames.

### Interactive Video Question Answering

Ask questions about objects appearing in the video.

Example:

```
Is there a knife in the video?
```

### Spatial Graph Video

Generate spatial scene graphs for selected frames.

### Semantic Graph Video

Extract semantic relations and captions for each frame.

---

# System Architecture

The pipeline integrates multiple components:

```
Input (Image / Video)
        │
        ▼
Object Detection (YOLO)
        │
        ▼
LION Multimodal Model
        │
        ├── Caption Generation
        ├── Question Answering
        ├── Semantic Tags
        │
        ▼
Scene Graph Generation
        │
        ├── Spatial Graph
        └── Semantic Graph
        │
        ▼
Symbolic Reasoning
        │
        ▼
Prolog Representation
```

---

# Installation

Clone the repository:

```
git clone https://github.com/castro2123/LION-PIII
cd LION-PIII
```

Create environment:

```
conda create -n lion python=3.12
conda activate lion
pip install -r requirements.txt
```

---

# Required Models

Download the following pretrained models:

## Checkpoints

| Version | Checkpoint |
| --- | --- |
| LION-FlanT5-XL| [daybreaksly/LION-FlanT5-XL](https://huggingface.co/daybreaksly/LION-FlanT5-XL) |


### Prepare models

1. Download the pre-trained vit model [eva_vit_g](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth).
2. Download the pre-trained RAM model [ram_swin_large_14m](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth).
3. Download the pre-trained FlanT5 model [FlanT5-XL](https://huggingface.co/google/flan-t5-xl).
4. Download the pre-trained BERT model [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
5. Fill in the paths to these models into the corresponding locations in the config file `configs\models\lion_flant5xl.yaml`


Comando de download do pre-trained FlanT5 model [FlanT5-XL]:
```
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/flan-t5-xl', local_dir='checkpoints/flan-t5-xl', local_dir_use_symlinks=False)"
```
Comando de download pre-trained BERT model [bert-base-uncased]:
```
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bert-base-uncased', local_dir='checkpoints/bert-uncased', local_dir_use_symlinks=False)"
```

# Running the Application

Launch the Streamlit interface:

```
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# Supported Modes

| Mode                  | Description                     |
| --------------------- | ------------------------------- |
| Caption               | Generate image descriptions     |
| Bounding Box          | Detect objects                  |
| Clustering            | Group detected objects          |
| Interactive QA        | Ask questions about images      |
| Spatial Graph         | Generate spatial relations      |
| Semantic Graph        | Extract semantic relations      |
| Prolog Representation | Logical reasoning               |
| Video Caption         | Caption key frames              |
| Bounding Box Video    | Object detection in videos      |
| Clustering Video      | Object clustering across frames |
| Interactive Video QA  | Ask questions about videos      |
| Spatial Graph Video   | Spatial relations in frames     |
| Semantic Graph Video  | Semantic relations in frames    |

---

# Example Interface

The application provides a simple UI:

1. Upload an **image or video**
2. Select the **analysis mode**
3. Run the selected task
4. Visualize results interactively

---

# Research Background

This project builds upon the paper:

**LION: Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge**

Presented at **CVPR 2024**.

Paper:
https://arxiv.org/abs/2311.11860

---

# Citation

If you use this project, please cite:

```
@inproceedings{chen2024lion,
title={LION: Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge},
author={Chen, Gongwei and Shen, Leyang and Shao, Rui and Deng, Xiang and Nie, Liqiang},
booktitle={CVPR},
year={2024}
}
```


# License

This project follows the same license as the original LION repository.
