export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

deepspeed --master-port 29510 --include localhost:0,1 gen_moses.py --dist 1 \
                                            --gpu 0 \
                                            --batch_size 20  \
                                            --exp_name moses \
                                            --exp_id 0 \
                                            --return_num 100    \
                                            --max_len 55   \
                                            --min_len 13    \
                                            --top_k 30  \
                                            --top_p 1   \
                                            --beam 100  \
                                            --process 'preprocess'  \
                                            --generate_mode 'topk'  \
                                            --checkpoint_path '../moldata/checkpoint/molgen.pkl' \
                                            --input_path '../moldata/finetune/moses_test.csv'  \
                                            --generate_path '../moldata/generate/gen_moses.csv' \
                                            --deepspeed \
                                            --deepspeed_config generate_config.json \
                                            --wte_path                  '../moldata/states/wte.pkl'                 \
                                            --control_trans_path        '../moldata/states/control_trans.pkl'       \
                                            --wte_enc_path              '../moldata/states/wte_enc.pkl'             \
                                            --control_trans_enc_path    '../moldata/states/control_trans_enc.pkl'   \
                                            --wte_dec_path              '../moldata/states/wte_dec.pkl'             \
                                            --control_trans_dec_path    '../moldata/states/control_trans_dec.pkl'   \
