import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import backend as K
import torch
from transformers import BertTokenizer, BertModel
import esm

# -----------------------------
# Model definition
# -----------------------------
def ptcr_fusenet():
    dropout_rate = 0.2
    ac = tf.nn.swish

    # Peptide branch
    input_pep = Input(shape=(1024 + 320,), name='peptide')
    x_pep = layers.Dense(256, activation=ac)(input_pep)
    x_pep = layers.BatchNormalization()(x_pep)
    x_pep = layers.Dropout(dropout_rate)(x_pep)

    # TCR alpha branch
    input_tcra = Input(shape=(1024 + 320,), name='tcra')
    x_tcra = layers.Dense(512, activation=ac)(input_tcra)
    x_tcra = layers.BatchNormalization()(x_tcra)
    x_tcra = layers.Dropout(dropout_rate)(x_tcra)
    x_tcra = layers.Dense(256, activation=ac)(x_tcra)
    x_tcra = layers.BatchNormalization()(x_tcra)
    x_tcra = layers.Dropout(dropout_rate)(x_tcra)
    x_tcra = layers.Dense(128, activation=ac)(x_tcra)
    x_tcra = layers.BatchNormalization()(x_tcra)
    x_tcra = layers.Dropout(dropout_rate)(x_tcra)

    # TCR beta branch
    input_tcrb = Input(shape=(1024 + 320,), name='tcrb')
    x_tcrb = layers.Dense(512, activation=ac)(input_tcrb)
    x_tcrb = layers.BatchNormalization()(x_tcrb)
    x_tcrb = layers.Dropout(dropout_rate)(x_tcrb)
    x_tcrb = layers.Dense(256, activation=ac)(x_tcrb)
    x_tcrb = layers.BatchNormalization()(x_tcrb)
    x_tcrb = layers.Dropout(dropout_rate)(x_tcrb)
    x_tcrb = layers.Dense(128, activation=ac)(x_tcrb)
    x_tcrb = layers.BatchNormalization()(x_tcrb)
    x_tcrb = layers.Dropout(dropout_rate)(x_tcrb)

    x_tcr = layers.Concatenate()([x_tcra, x_tcrb])

    diff_func = lambda tensors: K.abs(tensors[0] - tensors[1])
    diff = layers.Lambda(diff_func, output_shape=(256,))([x_tcr, x_pep])

    mul = layers.Multiply()([x_tcr, x_pep])

    x = layers.Concatenate()([x_tcr, x_pep, diff, mul])
    x = layers.Dense(256, activation=ac)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation=ac)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation=ac)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_pep, input_tcra, input_tcrb],
                  outputs=output,
                  name="pTCR-FuseNet")
    return model


# -----------------------------
# Embedding utilities
# -----------------------------
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device)

def get_esm_embedding_batch(seqs):
    batch_labels = [("sequence", seq) for seq in seqs]
    batch_tokens = batch_converter(batch_labels)[2]
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)

    token_representations = results["representations"][6]
    embeddings = []
    for i, (_, seq) in enumerate(batch_labels):
        emb = token_representations[i, 1:len(seq)+1].mean(0).cpu().numpy()
        embeddings.append(emb)
    return embeddings


tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
bert_model = bert_model.eval().to(device)

def add_space_between_amino_acids(seq):
    return ' '.join(seq)

def get_protbert_embedding(sequence):
    sequence = add_space_between_amino_acids(sequence)
    tokens = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True)
    input_ids = tokens['input_ids'].to(bert_model.device)
    attention_mask = tokens['attention_mask'].to(bert_model.device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    embedding = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
    return embedding.squeeze().cpu().numpy()

def extract_combined_embeddings(df, esm_dict, protbert_dict, a_col='A', b_col='B', pep_col='peptide'):
    def get_combined(seq):
        esm_emb = esm_dict.get(seq)
        protbert_emb = protbert_dict.get(seq)
        if esm_emb is not None and protbert_emb is not None:
            return np.concatenate([esm_emb, protbert_emb])
        return None

    A_emb = np.stack(df[a_col].map(get_combined).values)
    B_emb = np.stack(df[b_col].map(get_combined).values)
    pep_emb = np.stack(df[pep_col].map(get_combined).values)
    return pep_emb.astype(np.float32), A_emb.astype(np.float32), B_emb.astype(np.float32)


# -----------------------------
# Main prediction script
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("case_study.csv")
    df['A'] = df['CDR1_Alpha'] + 'X' + df['CDR2_Alpha'] + 'X' + df['CDR3_Alpha']
    df['B'] = df['CDR1_Beta'] + 'X' + df['CDR2_Beta'] + 'X' + df['CDR3_Beta']
    df.rename(columns={'Peptide_Sequence': 'peptide', 'Label': 'label'}, inplace=True)

    sequences = pd.Series(pd.concat([df['A'], df['B'], df['peptide']]).unique())

    all_embeddings, protbert_embeddings = {}, {}
    print("Generating embeddings...")
    for seq in sequences:
        all_embeddings[seq] = get_esm_embedding_batch([seq])[0]
        protbert_embeddings[seq] = get_protbert_embedding(seq)

    pep_emb, A_emb, B_emb = extract_combined_embeddings(df, all_embeddings, protbert_embeddings)

    model = ptcr_fusenet()
    model.load_weights("pTCR_FuseNet_pretrained_model.h5")

    y_pred_proba = model.predict([pep_emb, A_emb, B_emb])
    df['pred_proba'] = y_pred_proba
    df['pred_label'] = (df['pred_proba'] >= 0.5).astype(int)

    df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
