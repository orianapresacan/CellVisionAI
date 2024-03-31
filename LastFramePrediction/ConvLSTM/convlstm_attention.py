import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, device):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([x, h_cur], dim=1)  # Concatenate along channel axis
        cc_i, cc_f, cc_o, cc_g = torch.split(self.conv(combined), self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, height, width, attention_dim=256):
        super(SelfAttention, self).__init__()
        self.scale = 1. / attention_dim ** 0.5

        self.query = nn.Conv2d(hidden_dim, attention_dim, 1)
        self.key = nn.Conv2d(hidden_dim, attention_dim, 1)
        self.value = nn.Conv2d(hidden_dim, attention_dim, 1)

    def forward(self, x):
        batch_size, seq_len, hidden_dim, height, width = x.size()

        # Reshape x to apply convolutional attention: merge batch and seq_len dimensions
        x_reshaped = x.view(batch_size * seq_len, hidden_dim, height, width)

        # Apply convolutional layers to get query, key, and value
        query = self.query(x_reshaped)
        key = self.key(x_reshaped)
        value = self.value(x_reshaped)

        # Reshape back to separate batch and seq_len dimensions
        query = query.view(batch_size, seq_len, -1, height * width)
        key = key.view(batch_size, seq_len, -1, height * width)
        value = value.view(batch_size, seq_len, -1, height * width)

        # Transpose and reshape for attention calculation
        query = query.permute(0, 3, 2, 1).contiguous().view(batch_size * height * width, seq_len, -1)
        key = key.permute(0, 3, 1, 2).contiguous().view(batch_size * height * width, -1, seq_len)
        value = value.permute(0, 3, 2, 1).contiguous().view(batch_size * height * width, seq_len, -1)

        # Calculate attention
        attention_scores = torch.bmm(query, key) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.bmm(attention_weights, value)

        # Reshape attended back to original spatial dimensions
        attended = attended.view(batch_size, height, width, -1, seq_len)
        attended = attended.permute(0, 4, 3, 1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, -1, height, width)

        return attended



class ConvLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, n_layers, device, batch_first=False, return_sequences=False):
        super(ConvLSTM_Attention, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_dims = hidden_dims
        self.return_sequences = return_sequences

        for i in range(n_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.layers.append(ConvLSTMCell(input_dim=cur_input_dim,
                                            hidden_dim=hidden_dims[i],
                                            kernel_size=kernel_sizes[i],
                                            device=device))

        flatten_dim = 128 * 64 * 64  # hidden_dims[-1] * output_height * output_width
        # hidden_dims = [128, 128, 128, 128]

        height = 64
        width = 64

        self.self_attention = SelfAttention(hidden_dims[-1], height, width)
        # self.final_conv = nn.Conv2d(hidden_dims[-1], 3, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, _, height, width = x.size()
        hidden_states = [layer.init_hidden(batch_size, (height, width), x.device) for layer in self.layers]

        output_sequence = []

        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            for layer_idx, layer in enumerate(self.layers):
                h, c = layer(x_t, hidden_states[layer_idx])
                x_t = h
                hidden_states[layer_idx] = (h, c)

            output_sequence.append(x_t)

        output_sequence = torch.stack(output_sequence, dim=1)

        # Apply self-attention
        attended_sequence = self.self_attention(output_sequence)

        # If not returning sequences, take the last time step
        if not self.return_sequences:
            attended_sequence = attended_sequence[:, -1, :, :, :]

        # Final Convolution
        # Ensure attended_sequence is 4D before applying convolution
        if attended_sequence.dim() == 4:
            output = self.final_conv(attended_sequence)
        else:
            raise RuntimeError(f"Expected 4D tensor for convolution, but got tensor of size: {attended_sequence.size()}")

        return output