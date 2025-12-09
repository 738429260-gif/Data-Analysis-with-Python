import numpy as np


def softmax(vector):
    """
    vector: np.array of shape (n, m)

    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    """
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_


def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    # scores: shape (1, n_states)
    # e_i = s^T W_mult h_i
    scores = decoder_hidden_state.T.dot(W_mult).dot(encoder_hidden_states)

    # attention weights (1, n_states)
    softmax_vector = softmax(scores)

    # weighted sum of encoder states → (n_features_enc, 1)
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T

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
    # project encoder states: (n_features_int, n_states)
    enc_part = W_add_enc.dot(encoder_hidden_states)

    # project decoder state: (n_features_int, 1)
    dec_part = W_add_dec.dot(decoder_hidden_state)

    # broadcast decoder projection over all time steps and apply tanh
    # result: (n_features_int, n_states)
    activation = np.tanh(enc_part + dec_part)

    # scores: (1, n_states)
    scores = v_add.T.dot(activation)

    # attention weights: (1, n_states)
    softmax_vector = softmax(scores)

    # weighted sum of encoder states → (n_features_enc, 1)
    attention_vector = softmax_vector.dot(encoder_hidden_states.T).T

    return attention_vector
