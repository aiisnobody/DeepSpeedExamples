# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
MASTER_ADDR=192.168.3.152
MASTER_PORT=29500
RANK=0

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

export DEEPSPEED_VERBOSE=1
export DEEPSPEED_DEBUG=1
#export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=^br-c485a8390817,docker0,eno2,ens6f0,ens6f1,enx46a838614d5f,lo,veth23d7383,br-dcd3e4ec14e7,enp194s0f1,ens6f0,ens6f1,enxe278666d5a52,veth110d0b7,veth215ea4e,veth3203d6b,veth87c3cbf,vethec6fc79,virbr0


export NCCL_IB_DISABLE=1 && deepspeed --hostfile hostfile --no_ssh --num_gpu 4 --num_nodes 2 --master_addr $MASTER_ADDR --master_port 12344 --node_rank=0 main.py \
   --data_path yitingxie/rlhf-reward-datasets \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-66b \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --pipeline_parallel_size 2 \
   --model_parallel_size 2 \
   --output_dir $OUTPUT \
