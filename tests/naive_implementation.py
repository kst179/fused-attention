import torch


def attention(queries, keys, values, head_dim):
    batch_size, sequence_len, num_features = queries.shape
    num_heads = num_features // head_dim

    queries = queries.view(batch_size, sequence_len, num_heads, head_dim).transpose(1, 2)
    keys = keys.view(batch_size, sequence_len, num_heads, head_dim).transpose(1, 2)
    values = values.view(batch_size, sequence_len, num_heads, head_dim).transpose(1, 2)

    scores = (queries @ keys.mT) / (head_dim ** 0.5)
    scores = torch.softmax(scores, dim=-1)
    output = scores @ values

    output = output.transpose(1, 2).contiguous().view(batch_size, sequence_len, num_features)    

    return output