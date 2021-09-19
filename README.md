# JD2Skills-BERT-XMLC
### [Dataset](https://github.com/WING-NUS/JD2Skills-BERT-XMLC/blob/main/data/mycareersfuture.tar.gz) | [Paper](https://github.com/WING-NUS/JD2Skills-BERT-XMLC/blob/main/doc/COLING_2020.pdf) | [PPT](https://github.com/WING-NUS/JD2Skills-BERT-XMLC/blob/main/doc/COLING2020.pptx) | [Presentation](https://youtu.be/fqHvYIEh-Q8)
Code and Dataset for the Bhola et al. (2020) **Retrieving Skills from Job Descriptions: A Language Model Based Extreme Multi-label Classification Framework**

Default model weights and dataset are available in the [link](https://drive.google.com/drive/folders/1e2IwSbF5DTk-3OQITBxlYOQGRISL2DBV?usp=sharing)
## Dataset

The dataset is collected from Singaporean government website, mycareersfuture.sg consisting of over 20, 000 richly-structured job posts. The detailed statistics of the dataset are shown below:

Mycareersfuture.sg Dataset |  Stats
--- | ---
Number of job posts | 20,298
Number of distinct skills | 2,548
Number of skills with 20 or more mentions | 1,209
Average skill tags per job post | 19.98
Average token count per job post | 162.27
Maximum token count in a job post | 1,127


This dataset includes the following fields:

1. company_name
2. job_title
3. employment_type
4. seniority
5. job_category
6. location
7. salary
8. min_experience
9. skills_required
10. requirements_and_role
11. job_requirements
12. company_info
13. posting_date
14. expiry_date
15. no_of_applications
16. job_id


## BERT-XMLC model
The proposed model constitutes a pre-trained BERT based text encoder utilizing WordPiece embedding. The encoded textual representation is passed into bottleneck layer. This layer alleviates overfitting by (significantly) limiting the number of trainable parameters. The activations are passed through fully connected layer, finally producing probability scores using sigmoid activation function.

<p align="center">
  <img width="460"  src="https://github.com/WING-NUS/JD2Skills-BERT-XMLC/blob/main/doc/BERT-XMLC.png">
</p>

## Model setup
* Run ```bash setup.sh```

**Or**

* Transfer all files from ```checkpoint``` folder (in google drive) to ```pybert/pretrain/bert/bert-uncased``` folder
* Transfer dataset files from ```dataset``` folder (in google drive) to ```pybert/dataset``` folder

**Training** <br> 
``` python run_bert.py --train --data_name job_dataset ```

**Testing** <br> 
``` python run_bert.py --test --data_name job_dataset```

**Note:** Configurations for training, validation and testing of model are provided in ```pybert/configs/basic_config.py``` <br>
Additionally, ```pybert/model_setup/CAB_dataset_script.py``` is provided to implement CAB

## Results

Experimental results on skill prediction task are shown below:

<p align="center">
  <img width="400"  src="https://github.com/WING-NUS/JD2Skills-BERT-XMLC/blob/main/doc/Screenshot%202020-11-30%20163740.png">
</p>

<p align="center">
  <img width="600"  src="https://github.com/WING-NUS/JD2Skills-BERT-XMLC/blob/main/doc/Screenshot%202020-11-30%20163812.png">
</p>

<p align="center">
  <img width="600"  src="https://github.com/WING-NUS/JD2Skills-BERT-XMLC/blob/main/doc/Screenshot%202020-11-30%20163845.png">
</p>

**Note:** Model has been further finetuned

## Bibtex
```
@inproceedings{bhola-etal-2020-retrieving,
    title = "Retrieving Skills from Job Descriptions: A Language Model Based Extreme Multi-label Classification Framework",
    author = "Bhola, Akshay  and
      Halder, Kishaloy  and
      Prasad, Animesh  and
      Kan, Min-Yen",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.513",
    doi = "10.18653/v1/2020.coling-main.513",
    pages = "5832--5842",
    abstract = "We introduce a deep learning model to learn the set of enumerated job skills associated with a job description. In our analysis of a large-scale government job portal mycareersfuture.sg, we observe that as much as 65{\%} of job descriptions miss describing a significant number of relevant skills. Our model addresses this task from the perspective of an extreme multi-label classification (XMLC) problem, where descriptions are the evidence for the binary relevance of thousands of individual skills. Building upon the current state-of-the-art language modeling approaches such as BERT, we show our XMLC method improves on an existing baseline solution by over 9{\%} and 7{\%} absolute improvements in terms of recall and normalized discounted cumulative gain. We further show that our approach effectively addresses the missing skills problem, and helps in recovering relevant skills that were missed out in the job postings by taking into account the structured semantic representation of skills and their co-occurrences through a Correlation Aware Bootstrapping process. We further show that our approach, to ensure the BERT-XMLC model accounts for structured semantic representation of skills and their co-occurrences through a Correlation Aware Bootstrapping process, effectively addresses the missing skills problem, and helps in recovering relevant skills that were missed out in the job postings. To facilitate future research and replication of our work, we have made the dataset and the implementation of our model publicly available.",
}
```
