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


class ConvLSTM_Michael(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, n_layers, device, batch_first=False, return_sequences=False):
        super(ConvLSTM_Michael, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.layers.append(ConvLSTMCell(input_dim=cur_input_dim,
                                            hidden_dim=hidden_dims[i],
                                            kernel_size=kernel_sizes[i],
                                            device=device))

        self.final_conv = nn.Conv2d(hidden_dims[-1], 3, kernel_size=1)  # Changed from 1 to 3
        self.return_sequences = return_sequences


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
            
            # If return_sequences, store the output of each time step
            if self.return_sequences:
                output_sequence.append(x_t)

        # Use the output of the last time step if return_sequences is False, else use the entire sequence
        x_t_final = x_t if not self.return_sequences else torch.stack(output_sequence, dim=1)

        # Reshape x_t_final for 4D convolution
        if self.return_sequences:
            # Reshape keeping the sequence
            x_t_combined = x_t_final.view(batch_size * seq_len, self.layers[-1].hidden_dim, height, width)
        else:
            # Reshape without sequence dimension
            x_t_combined = x_t_final.view(batch_size, self.layers[-1].hidden_dim, height, width)

        output = self.final_conv(x_t_combined)

        # Reshape back to 5D if return_sequences is True
        if self.return_sequences:
            output = output.view(batch_size, seq_len, -1, height, width)

        return output


