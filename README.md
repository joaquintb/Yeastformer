# Yeastformer

**Yeastformer** is a transformer-based model inspired by the [Geneformer](https://www.nature.com/articles/s41586-023-06139-9) framework, with a focus on analyzing *Saccharomyces cerevisiae* (yeast) data.

This repository is currently focused on developing an effective tokenization strategy for dual-channel microarray data. While this aspect is still under exploration, the codebase already includes working scripts for pretraining the model, as well as tools for analyzing attention patterns and generating gene embeddings for pretrained models.

### Data

The yeast gene expression data used in this project were collected from the [Saccharomyces Genome Database (SGD)](http://sgd-archive.yeastgenome.org/expression/microarray/). Specifically, we used publicly available microarray datasets from the SGD expression archive.

### Set-Up

...

### Overview

...

### Repository Walkthrough 

* **data**
  * *dual_channel_pcls_modified*: folder containing the dual-channel microarray pcl fields that are preprocessed and merged to become the master matrix of expression data.
  * *genes_info*: folder containing complementary information about yeast genes.
    * *sample_hk_genes_list.pkl*: predefined list of yeast housekeeping genes.
    * *sample_tf_genes_list.pkl*: predefined list of yeast transcription factors.
    * *all_yeast_genes.tsv*: mapping between systematic and standard notation for each yeast gene in the genome. Needed to have the same type of identifier for each gene in the master matrix.
    * *all_yeast_genes_rest_of_problematic_update.tsv*: same document but including some manual addtions to deal with some specific cases manually.
  * *output:* folder collecting the outputs related to data (*e.g.* master matrix of data in csv format, token dictionary)
  * *pcl_preprocessing.ipynb*: processing expression files to ensure a succesful and sensible merge.
  * *data_inspection.ipynb*: exploring some initial questions about the data.
  * *hs_tf_study.ipynb*: exploring the behavior of housekeepings vs transcription factors in our data.
  * *merging.py:* script merging the preprocessed *.pcl* expression files from the different experiments into a single *.csv* file with gene identifiers as rows and the union of experimental conditions as columns.
  * *building_dataset.ipynb*: generating the final *.dataset* used to pretrain the model. This notebook uses the Geneformer approach to tokenizing, even using Geneformer's tokenizer. However, **this is the main issue under development, since an alternative tokenization may be needed for this problem; we are dealing with dual-channel microarray data instead of single-cell data.**
