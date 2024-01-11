# sscNOVA

Welcome to the ``` sscNOVA ``` framework repository!
``` sscNOVA ``` is a semi-supervised convolutional neural network algorithm to identify functional regulatory variants from GWAS and eQTLs dataset and exploring the functional characteristics of regulatory variants in autoimmune diseases. 

# Requirements

Please create a new conda environment specifically for running ``` sscNOVA ``` (e.g. ``` conda create --sscNOVA python=3.10.9 ```), install the packages listed in the ``` requirements.txt ``` file. Install with conda or pip (e.g. ``` conda install pandas==2.0.3 ```).

# Data

Autoimmune disease-related data are downloaded from the GWAS Catalog with GRCh38 human reference genome 
(https://www.ebi.ac.uk/gwas/docs/file-downloads), the version we are using is "All associations v1.0". 
The Immune Cell Gene Expression Atlas from the University of Tokyo (ImmuNexUT) data are downloaded from National Bioscience Database Centre (NBDC) Human Database (https://humandbs.biosciencedbc.jp/en/hum0214-v6), the accession number E-GEAD-398 and E-GEAD-420 are used.

# Predicting

The following scripts can be used to obtain sscNOVA predictions for variants.

Example usage:

```
sh predict.sh <vcf_file> <hg_version> <output_dir> [--cuda] <output_file>
```

Arguments:

* ```<vcf_file>```: VCF file

* ```<hg_version>```: Either hg19 or hg38

* ```<output_dir>```: Path to feature annotation output directory.

* ```--cuda```: Optional, use this flag if running on a CUDA-enabled GPU.

* ```<output_file>```: Path to the prediction probability output file.

The output will be saved to output_file, the first five columns of the output file will be the same as the vcf files, the additional columns include predicted probability for each variant.

# Training

You can train the model through the following process.

### Feature annotation

Feature annotation, please refer to Chen KM, Wong AK, Troyanskaya OG, et al. "A sequence-based global map of regulatory activity for deciphering human genetics" (https://github.com/FunctionLab/sei-framework).

To obtain 21,907 chromatin profile features and 40 sequence class features for each variant, follow and execute the following two commands mentioned in the Sei repository's README.md:

``` sh 1_variant_effect_prediction.sh <vcf> <hg> <output-dir> [--cuda] ```

``` sh 2_varianteffect_sc_score.sh <ref-fp> <alt-fp> <output-dir> [--no-tsv] ```

### Feature selection

Annotating variants features, extracting corresponding features, and subsequently tagging variants.
 
Example usage:

```
python 01_feature_extraction.py feature_141.txt gwas_immunexut/sorted.gwas_immunexut.chromatin_profile_diffs.tsv gwas_immunexut_141.csv 1
```

Arguments:

python your_script_name.py feature_filename.txt input_filename.csv output_filename.csv label

### Training supervised model

After annotating the features, train a supervised model.

Example usage:

```
python 02_cnn.py gwas_immunexut_141.csv 141
```

Arguments:

python your_script_name.py input_filename.csv feature

### Predicting GWAS without ImmuNexUT interaction data

Use the trained model to predict GWAS without ImmuNexUT interaction data.

Example usage:

```
python 03_variant_predict_cnn.py gwas_remaining_141.csv cnn_141
```

Arguments:

python your_script_name.py input_filename.csv model_name

### Pseudo-labelling

Apply pseudo-labeling to GWAS without ImmuNexUT interaction data.

Example usage:

```
python 04_t_test.py gwas_remaining_141_cnn_141_probability.csv 0.9 0.1
```

Arguments:

python your_script_name.py input_filename.csv max_threshold min_threshold

### The optimal threshold of pseudo-labelling

The optimal threshold for applying pseudo-labels with different models.

| Model | Feature | Label:Threshold | Positive | Negative | Abandon |
| ----- | ----- | ----- | ----- | ----- | ----- |
| SVM | 40 | 1: >0.9, 0: <0.1 | 666 | 0 | 6258 |
| SVM | 150 | 1: >0.7, 0: <0.3 | 1912 | 0 | 5012 |
| SVM | 141 | 1: >0.6, 0: <0.4 | 2439 | 3916 | 569 |
| RF | 40 | 1: >0.9, 0: <0.1 | 808 | 432 | 5684 |
| RF | 150 | 1: >0.9, 0: <0.1 | 1068 | 555 | 5301 |
| RF | 141 | 1: >0.9, 0: <0.1 | 1505 | 570 | 4849 |
| CNN | 40 | 1: >0.5, 0: <0.5 | 3286 | 1987 | 1651 |
| CNN | 150 | 1: >0.6, 0: <0.4 | 3706 | 2463 | 755 |
| CNN | 141 | 1: >0.9, 0: <0.1 | 2759 | 626 | 3539 |
| TF | 40 | 1: >0.9, 0: <0.1 | 1896 | 648 | 4380 |
| TF | 150 | 1: >0.5, 0: <0.5 | 3608 | 2534 | 782 |
| TF | 141 | 1: >0.8, 0: <0.2 | 3420 | 2166 | 1338 |

### Training semi-supervised model

Train a semi-supervised model after applying pseudo-labels.

Example usage:

```
python 05_cnn_pseudo.py gwas_immunexut_141.csv gwas_remaining_141_cnn_141_probability_label_0.9_0.1.csv 141
```

Arguments:

python your_script_name.py input_filename.csv input_filename2.csv feature

### Predicting HGMD & ClinVar data

Predict an experimentally curated testing data.

Example usage:

```
python 06_variant_predict_indicator.py hgmd_clinvar_141.csv cnn_pseudo_141 141
```

Arguments:

python your_script_name.py input_filename.csv model_name feature

# Help

Please post in the Github issues or e-mail Fangyuan Shi (shify@nxu.edu.cn) with any questions about the repository, requests for more data, additional information about the results, etc.




