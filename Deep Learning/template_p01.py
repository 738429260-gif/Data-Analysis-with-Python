import numpy as np


def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # e_i = s^T W_mult h_i  →  all scores in one matrix:
    # decoder_hidden_state.T: (1, n_features_dec)
    # W_mult:                 (n_features_dec, n_features_enc)
    # encoder_hidden_states:  (n_features_enc, n_states)
    scores = decoder_hidden_state.T.dot(W_mult).dot(encoder_hidden_states)  # (1, n_states)

    # Convert scores to attention weights with softmax
    softmax_vector = softmax(scores)  # (1, n_states)

    # Weighted sum of encoder states
    # softmax_vector:        (1, n_states)
    # encoder_hidden_states: (n_features_enc, n_states)
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T  # (n_features_enc, 1)

    return attention_vector


def additive_attention(decoder_hidden_state, encoder_hidden_states,
                       v_add, W_add_enc, W_add_dec):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # Project encoder states: (n_features_int, n_states)
    enc_part = W_add_enc.dot(encoder_hidden_states)

    # Project decoder state: (n_features_int, 1)
    dec_part = W_add_dec.dot(decoder_hidden_state)

    # Broadcast decoder projection across all time steps and apply tanh
    # result shape: (n_features_int, n_states)
    activation = np.tanh(enc_part + dec_part)

    # e_i = v_add^T * tanh(...)
    # v_add.T:   (1, n_features_int)
    # activation: (n_features_int, n_states)
    scores = v_add.T.dot(activation)  # (1, n_states)

    # Convert scores to attention weights
    softmax_vector = softmax(scores)  # (1, n_states)

    # Weighted sum of encoder states
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T  # (n_features_enc, 1)

    return attention_vector
