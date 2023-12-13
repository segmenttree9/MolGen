export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

deepspeed --include localhost:5,6 train_moses.py --dist 1 \
                                --rank 0 \
                                --gpu 2 \
                                --batch_size 200 \
                                --exp_name train_moses \
                                --exp_id 001 \
                                --pretrain_path '../moldata/pretrain/train_first_200K.csv' \
                                --checkpoint_path '../moldata/checkpoint/molgen.pkl' \
                                --epoch 100 \
                                --workers 4 \
                                --accumulation_steps 8 \
                                --lr 1e-5 \
                                --deepspeed \
                                --deepspeed_config config.json \
                                --weight_decay 0.0001 \
                                --eval_step 20 \
                                --wte_path                  '../moldata/states/wte.pkl'                 \
                                --control_trans_path        '../moldata/states/control_trans.pkl'       \
                                --wte_enc_path              '../moldata/states/wte_enc.pkl'             \
                                --control_trans_enc_path    '../moldata/states/control_trans_enc.pkl'   \
                                --wte_dec_path              '../moldata/states/wte_dec.pkl'             \
                                --control_trans_dec_path    '../moldata/states/control_trans_dec.pkl'   \
                                