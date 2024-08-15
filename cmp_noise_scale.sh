cd main_code

export GPU_IDS="0,1,2,3"
export NUM_TEST=1024
export NUM_GPU=4
export BATCH_SIZE=32
export CLSSIFIER=t7



CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.1 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.2 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.3 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.4 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.5 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.6 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.7 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.8 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.9 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_linf.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 1.0 --model_type $CLSSIFIER


CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.1 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.2 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.3 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.4 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.5 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.6 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.7 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.8 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 0.9 --model_type $CLSSIFIER
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38346 cm_attack_eval_l2.py  --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1 --sigma_max 1.0 --model_type $CLSSIFIER