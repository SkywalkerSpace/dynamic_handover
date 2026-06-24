import torch
import torch.nn as nn

"""RNN modules."""


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        self.rnn = nn.GRU(inputs_dim, outputs_dim,
                          num_layers=self._recurrent_N)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        # 如果当前输入的批量大小与隐状态的批量大小一致，意味着当前是在进行环境 Rollout 采样或单步前向推理
        if x.size(0) == hxs.size(0):
            # 将隐状态乘以对应的 masks (若 mask 为 0，表明环境已重置，清除 RNN 的历史隐状态)
            x, hxs = self.rnn(x.unsqueeze(0),
                              (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # 否则是在进行策略网络训练更新：此时输入数据是折叠的序列数据 (T * N, -1)
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten：重新展开为时间步长与批次大小形式的张量 (T, N, -1)
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # 寻找序列中哪些时间步被终止 (masks 中存在 0)，并将时序划分为多个无重置的子片段进行高效的并行 BPTT 计算
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # 针对不包含重置信号的子时序片段，将其作为一个整体输入 GRU 运行前向更新，显著提高计算速度
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = (hxs * masks[start_idx].view(1, -1,
                        1).repeat(self._recurrent_N, 1, 1)).contiguous()
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            # 合并所有子时序片段的隐藏状态输出
            x = torch.cat(outputs, dim=0)

            # 重新压平为 (T * N, -1) 维度形式
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)

        # 增加 LayerNorm 层进行隐藏层特征归一化，提升神经网络的泛化性和训练稳定性
        x = self.norm(x)
        return x, hxs
