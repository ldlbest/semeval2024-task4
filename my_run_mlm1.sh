python my_run_mlm.py \
    --do_eval False \
    --learning_rate=3e-05 \
    --weight_decay=0.01 \
    --model_name_or_path ./pretrained_models/roberta-base \
    --train_file ./data/datasets/train-articles \
    --per_device_train_batch_size 16 \
    --do_train true\
    --do_eval false \
    --overwrite_output_dir true \
    --logging_steps=40 \
    --output_dir ./output/roberta-base-mlm \
    --num_train_epochs=10.0\
    --evaluation_strategy epoch \
    --line_by_line true \
    --pad_to_max_length true\
    --save_total_limit 5