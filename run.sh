CUDA_VISIBLE_DEVICES=0 python3 train.py --agent BERT_sl --dataset_path data/xiaowei/neg/ --dataset xiaowei_neg --gradient_accumulation_steps 2 2>&1 | tee -a BERT_sl_neg.log

