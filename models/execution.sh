export CUDA_VISIBLE_DEVICES="0,1"

DATASET="HOVER"
MODEL="flan-t5-xl"
SETTING="open-book"
PROGRAM_FILE_NAME="Mixtral_NULL_HOVER_N=3_Hops=2_programs.json"
CACHE_DIR="../"

python ./models/program_execution.py \
    --dataset_name ${DATASET} \
    --setting ${SETTING} \
    --FV_data_path ../datasets \
    --program_dir ../results/programs \
    --program_file_name ${PROGRAM_FILE_NAME} \
    --corpus_index_path ../datasets/${DATASET}/corpus/index \
    --num_retrieved 5 \
    --max_evidence_length 4096 \
    --num_eval_samples -1 \
    --model_name google/${MODEL} \
    --output_dir ../results/fact_checking \
    --cache_dir ${CACHE_DIR}
