final_anchor="-best"
out_anchor="-fine-tune"
for f in roberta-base deberta-v3; do
    python main.py \
        --pretrained_lm ${f} \
        --final_model ${f}${final_anchor} \
        --output_dir ${f}${out_anchor}\
        --result_dir ${f}${final_anchor} 
done