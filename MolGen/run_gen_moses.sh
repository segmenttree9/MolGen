export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

deepspeed --master_port 29510 --include localhost:6,7 gen_moses.py --dist 1 \
                                            --gpu 2 \
                                            --batch_size 20  \
                                            --exp_name moses \
                                            --exp_id 0 \
                                            --return_num 10    \
                                            --max_len 55   \
                                            --min_len 13    \
                                            --top_k 30  \
                                            --top_p 1   \
                                            --beam 10  \
                                            --process 'preprocess'  \
                                            --generate_mode 'topk'  \
                                            --checkpoint_path '../moldata/checkpoint/200K_b200e100_00219.pkl' \
                                            --input_path '../moldata/pretrain/test_first_1K.csv'  \
                                            --output_path '../moldata/generate/200K_b200e100_00219_3.csv' \
                                            --deepspeed \
                                            --deepspeed_config generate_config.json \
                                            --wte_path                  '../moldata/states/wte.pkl'                 \
                                            --control_trans_path        '../moldata/states/control_trans.pkl'       \
                                            --wte_enc_path              '../moldata/states/wte_enc.pkl'             \
                                            --control_trans_enc_path    '../moldata/states/control_trans_enc.pkl'   \
                                            --wte_dec_path              '../moldata/states/wte_dec.pkl'             \
                                            --control_trans_dec_path    '../moldata/states/control_trans_dec.pkl'   \
