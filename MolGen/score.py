from pandarallel import pandarallel
import selfies as sf
import pandas as pd
import moses
import moses.metrics as mcs
import pprint
import os

moses_test_path = "../moldata/pretrain/test_first_10K.csv"
generate_moses_path = "../moldata/generate/200K_b200e100_00219_2.csv"
moses_train_path = "../moldata/pretrain/train_first_200K.csv"

def sf_decode(selfies):
    try:
        decode = sf.decoder(selfies)
        return decode
    except sf.DecoderError:
        print("Can't decode!")
        return ''

def moses_metric():    
    test = pd.read_csv(moses_test_path)['smiles'].tolist()
    df_gen = pd.read_csv(generate_moses_path)
    train = pd.read_csv(moses_train_path)['smiles'].tolist()
    pandarallel.initialize(shm_size_mb=60720, nb_workers=20, progress_bar=False)
    df_gen['candidate_smiles'] = df_gen['candidates'].parallel_apply(sf_decode)
    gen = df_gen['candidate_smiles'].tolist()
    metrics = moses.get_all_metrics(gen, n_jobs=20, device='cuda:6', batch_size=1024, test=test, test_scaffolds=test, train=train)
    
    pprint.pprint(generate_moses_path)
    pprint.pprint(metrics, sort_dicts=False)

def moses_metric_my(exceptFCD = False):
    test = pd.read_csv(moses_test_path)['smiles'].tolist()
    df_gen = pd.read_csv(generate_moses_path)
    train = pd.read_csv(moses_train_path)['smiles'].tolist()
    
    df_gen['candidate_smiles'] = df_gen['candidates'].apply(sf_decode)
    gen = df_gen['candidate_smiles'].tolist()
    metrics = {}
    if exceptFCD:
        metrics = moses.get_all_metrics_exceptFCD(gen, test=test, test_scaffolds=test, train=train)
    else:
        metrics = moses.get_all_metrics(gen, test=test, test_scaffolds=test, train=train)
    
    with open("./score.txt", "w") as f:
        f.write(pprint.pformat(metrics, sort_dicts=False))
        
    pprint.pprint(metrics, sort_dicts=False)

# moses_metric_my(exceptFCD=True)
moses_metric()