#python -m graphsage.unsupervised_train --train_prefix ./example_data/ppi --model graphsage_mean --max_total_steps 1000 --validate_iter 10
python -m graphsage.unsupervised_train --train_prefix ./json_graphs/left_train_thro/left_train --model graphsage_mean  --max_total_steps 1000000 --validate_iter 10 --model_name left_thro_rnd2 epochs 100
