# JD2Skills-BERT-XMLC
Code and Dataset for the Bhola et al. (2020) Retrieving Skills from Job Descriptions: A Language Model Based Extreme Multi-label Classification Framework

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
