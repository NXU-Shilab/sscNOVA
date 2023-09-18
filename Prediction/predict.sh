#!/bin/bash


# Check the number of input arguments
if [ $# -ne 5 ]; then
    echo "Usage: $0 <vcf_file> <hg_version> <output_dir> [--cuda] <feature_file>"
    exit 1
fi

# Get input files and parameters from command line arguments
vcf_file=$1
hg_version=$2
output_dir=$3
feature_file=$5
cuda=""

# Check if CUDA is enabled
if [ "$4" == "--cuda" ]; then
    cuda="--cuda"
fi

# 1. Run the first command and replace vcf_file and hg_version with external inputs
sh 1_variant_effect_prediction.sh $vcf_file $hg_version $output_dir $cuda

# Check the return status of the first command
if [ $? -ne 0 ]; then
    echo "The first command failed"
    exit 1
fi

# 2. Run the second command, replace other parameters with external inputs
sh 2_varianteffect_sc_score.sh "./$output_dir/chromatin-profiles-hdf5/$output_dir.ref_predictions.h5" "./$output_dir/chromatin-profiles-hdf5/$output_dir.alt_predictions.h5" "$output_dir"

# Check the return status of the second command
if [ $? -ne 0 ]; then
    echo "The second command failed"
    exit 1
fi

# 3. Run the third command, replace other parameters with external inputs
python feature_extraction.py "./feature.txt" "./$output_dir/sorted.$output_dir.chromatin_profile_diffs.tsv" $feature_file

# Check the return status of the third command
if [ $? -ne 0 ]; then
    echo "The third command failed"
    exit 1
fi

# 4. Run the fourth command, keep other parameters unchanged
python predict.py $feature_file
