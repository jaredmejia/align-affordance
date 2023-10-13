#!/bin/bash
# script to generate non-articulated object contact HO3-Pair data split

split_idx=$1

if (( split_idx < 0 || split_idx > 20 )); then
    echo "Error: split_idx must be between 0 and 20 (inclusive)"
    exit 1
fi

python generate_data.py --data_dir /scratch/jared/projects/affordances/human-affordances/aff-diff/data/ --save_dir /scratch/jared/projects/affordances/human-affordances/aff-diff/processed_data/non-art-contact-$split_idx --decode --inpaint --bbox --split docs/non_art_contact/non_art_contact_$split_idx.csv --num_workers 16 --bs 32

