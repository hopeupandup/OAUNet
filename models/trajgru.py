"""TrajGRU encoder-decoder with a heteroscedastic log-variance head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# log σ² is initialised at σ_u = 0.05 (≈ 4.8 dBZ), the order of the observed residuals.
_LOGVAR_INIT = math.log(0.05 ** 2)


def _run_cell(cell, use_checkpoint, x, h):
    if use_checkpoint and h is not None and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(
            cell, x, h, use_reentrant=False
        )
    return cell(x, h)


def create_grid(U, V, h, w):
    device = U.device
    x = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(-1, -1, h, w).to(device)
    y = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(-1, -1, h, w).to(device)

    x = x + U * 2
    y = y + V * 2

    grid = torch.cat([x, y], dim=1)
    grid = grid.permute(0, 2, 3, 1)

    return grid


def warp(H, U, V):
    B, C, h, w = H.shape

    grid = create_grid(U, V, h, w)

    warped = F.grid_sample(H, grid, mode='bilinear',
                           padding_mode='zeros', align_corners=True)

    return warped


class FlowGenerator(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_flows=9, kernel_size=3):
        super().__init__()

        self.num_flows = num_flows
        padding = kernel_size // 2

        self.fusion_conv = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

        self.flow_conv = nn.Conv2d(
            hidden_channels,
            2 * num_flows,
            kernel_size=kernel_size,
            padding=padding
        )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()

    def forward(self, x, h):
        concat = torch.cat([x, h], dim=1)

        fused = self.leaky_relu(self.fusion_conv(concat))

        flow_output = self.tanh(self.flow_conv(fused))

        flows = []
        for i in range(self.num_flows):
            U = flow_output[:, 2*i:2*i+1, :, :]
            V = flow_output[:, 2*i+1:2*i+2, :, :]
            flows.append((U, V))

        return flows


class TrajGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_flows=9, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_flows = num_flows
        padding = kernel_size // 2

        self.flow_generator = FlowGenerator(input_channels, hidden_channels, num_flows)

        self.conv_xz = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_hz_list = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
            for _ in range(num_flows)
        ])

        self.conv_xr = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_hr_list = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
            for _ in range(num_flows)
        ])

        self.conv_xh = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_hh_list = nn.ModuleList([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
            for _ in range(num_flows)
        ])

    def forward(self, x, h=None):
        B, _, H, W = x.shape

        if h is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)


        flows = self.flow_generator(x, h)


        warped_h_list = []
        for i, (U, V) in enumerate(flows):
            warped_h = warp(h, U, V)
            warped_h_list.append(warped_h)


        z = self.conv_xz(x)
        for i, warped_h in enumerate(warped_h_list):
            z = z + self.conv_hz_list[i](warped_h)
        z = torch.sigmoid(z)


        r = self.conv_xr(x)
        for i, warped_h in enumerate(warped_h_list):
            r = r + self.conv_hr_list[i](warped_h)
        r = torch.sigmoid(r)


        h_prime = self.conv_xh(x)
        for i, warped_h in enumerate(warped_h_list):
            h_prime = h_prime + r * self.conv_hh_list[i](warped_h)
        h_prime = torch.tanh(h_prime)


        h_new = (1 - z) * h_prime + z * h

        return h_new


class TrajGRUEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels_list, num_flows_list=None,
                 kernel_size=3, use_checkpoint=False):
        super().__init__()

        self.num_layers = len(hidden_channels_list)
        self.use_checkpoint = use_checkpoint

        if num_flows_list is None:
            num_flows_list = [9] * self.num_layers

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                cell = TrajGRUCell(
                    input_channels,
                    hidden_channels_list[i],
                    num_flows=num_flows_list[i],
                    kernel_size=kernel_size
                )
            else:
                cell = TrajGRUCell(
                    hidden_channels_list[i-1],
                    hidden_channels_list[i],
                    num_flows=num_flows_list[i],
                    kernel_size=kernel_size
                )
            self.cells.append(cell)

    def forward(self, x):
        B, T, C, H, W = x.shape

        all_hidden = []

        for layer_idx, cell in enumerate(self.cells):
            layer_hidden = []
            h = None

            for t in range(T):
                if layer_idx == 0:
                    inp = x[:, t, :, :, :]
                else:
                    inp = all_hidden[layer_idx - 1][:, t, :, :, :]

                h = _run_cell(cell, self.use_checkpoint, inp, h)

                layer_hidden.append(h)

            layer_hidden = torch.stack(layer_hidden, dim=1)
            all_hidden.append(layer_hidden)

        return all_hidden


class TrajGRUDecoder(nn.Module):
    def __init__(self, hidden_channels_list, output_channels, output_frames,
                 num_flows_list=None, kernel_size=3, use_checkpoint=False,
                 heteroscedastic=False):
        super().__init__()

        self.output_frames = output_frames
        self.num_layers = len(hidden_channels_list)
        self.use_checkpoint = use_checkpoint
        self.output_channels = output_channels
        self.heteroscedastic = bool(heteroscedastic)

        if num_flows_list is None:
            num_flows_list = [9] * self.num_layers

        self.cells = nn.ModuleList()
        for i in range(self.num_layers - 1, -1, -1):
            if i == self.num_layers - 1:
                cell = TrajGRUCell(
                    output_channels,
                    hidden_channels_list[i],
                    num_flows=num_flows_list[i],
                    kernel_size=kernel_size
                )
            else:
                cell = TrajGRUCell(
                    hidden_channels_list[i + 1],
                    hidden_channels_list[i],
                    num_flows=num_flows_list[i],
                    kernel_size=kernel_size
                )
            self.cells.append(cell)

        self.output_conv = nn.Sequential(
            nn.Conv2d(
                hidden_channels_list[0],
                output_channels,
                kernel_size,
                padding=kernel_size//2
            ),
            nn.ReLU()
        )

        # σ head: regresses log σ² per pixel. Without it the (y-μ)²/σ² term of the Gaussian NLL has
        # no σ able to explain the data noise, and the optimiser drives σ down to 1e-4 dBZ.


        if self.heteroscedastic:
            self.logvar_conv = nn.Conv2d(
                hidden_channels_list[0], output_channels,
                kernel_size, padding=kernel_size // 2
            )
            nn.init.zeros_(self.logvar_conv.weight)
            nn.init.constant_(self.logvar_conv.bias, _LOGVAR_INIT)

    def forward(self, encoder_hidden, teacher_forcing=None, return_logvar=False):
        B = encoder_hidden[0].shape[0]
        H = encoder_hidden[0].shape[3]
        W = encoder_hidden[0].shape[4]

        decoder_states = [hidden[:, -1, :, :, :] for hidden in encoder_hidden]
        decoder_states = decoder_states[::-1]

        prev_output = self.output_conv(decoder_states[-1])

        outputs = []
        logvars = [] if self.heteroscedastic else None

        for t in range(self.output_frames):
            new_states = []
            for layer_idx, cell in enumerate(self.cells):
                if layer_idx == 0:
                    inp = prev_output
                else:
                    inp = new_states[layer_idx - 1]

                h = _run_cell(cell, self.use_checkpoint, inp,
                              decoder_states[layer_idx])
                new_states.append(h)

            decoder_states = new_states

            output = self.output_conv(decoder_states[-1])
            outputs.append(output)
            if self.heteroscedastic:
                # The σ head reads the backbone state with detach(): NLL gradients never reach the shared trunk.
                logvars.append(self.logvar_conv(decoder_states[-1].detach()))

            if teacher_forcing is not None:
                prev_output = teacher_forcing[:, t, :, :, :]
            else:
                prev_output = output

        outputs = torch.stack(outputs, dim=1)
        if self.heteroscedastic and return_logvar:
            return outputs, torch.stack(logvars, dim=1)
        return outputs


class TrajGRU(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=[64, 128], num_layers=2,
                 output_frames=10, num_flows=[9, 9], kernel_size=3,
                 use_checkpoint=False, heteroscedastic=False):
        super().__init__()

        self.input_channels = input_channels
        self.output_frames = output_frames
        self.use_checkpoint = use_checkpoint
        self.heteroscedastic = bool(heteroscedastic)

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers

        if isinstance(num_flows, int):
            num_flows = [num_flows] * num_layers

        self.encoder = TrajGRUEncoder(
            input_channels=input_channels,
            hidden_channels_list=hidden_channels,
            num_flows_list=num_flows,
            kernel_size=kernel_size,
            use_checkpoint=use_checkpoint
        )

        self.decoder = TrajGRUDecoder(
            hidden_channels_list=hidden_channels,
            output_channels=input_channels,
            output_frames=output_frames,
            num_flows_list=num_flows,
            kernel_size=kernel_size,
            use_checkpoint=use_checkpoint,
            heteroscedastic=self.heteroscedastic
        )

    def set_use_checkpoint(self, flag):
        self.use_checkpoint = bool(flag)
        self.encoder.use_checkpoint = bool(flag)
        self.decoder.use_checkpoint = bool(flag)

    def forward(self, x, teacher_forcing=None, return_logvar=False):
        encoder_hidden = self.encoder(x)


        if getattr(self, 'stop_grad_encoder', False):
            encoder_hidden = [h.detach() for h in encoder_hidden]

        outputs = self.decoder(encoder_hidden, teacher_forcing,
                               return_logvar=return_logvar)

        return outputs


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数:")
    print(f"  总参数: {total:,}")
    print(f"  可训练: {trainable:,}")
    return total, trainable


def test_trajgru():
    print(f"\n{'='*60}")
    print(f"测试 TrajGRU 模型")
    print(f"{'='*60}")

    model = TrajGRU(
        input_channels=1,
        hidden_channels=[64, 128],
        num_layers=2,
        output_frames=10,
        num_flows=[9, 9]
    )

    count_parameters(model)

    print(f"\n测试前向传播:")
    x = torch.randn(2, 5, 1, 101, 101)
    print(f"  输入 shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        y = model(x)

    print(f"  输出 shape: {y.shape}")

    B, T_out, C, H, W = y.shape
    assert B == 2, f"Batch size 错误: 期望 2, 实际 {B}"
    assert T_out == 10, f"输出帧数错误: 期望 10, 实际 {T_out}"
    assert C == 1, f"输出通道错误: 期望 1, 实际 {C}"
    assert H == 101, f"高度错误: 期望 101, 实际 {H}"
    assert W == 101, f"宽度错误: 期望 101, 实际 {W}"

    print(f"  输出范围: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  输出均值: {y.mean():.4f}")

    print(f"\n测试教师强制:")
    teacher = torch.randn(2, 10, 1, 101, 101)
    with torch.no_grad():
        y_tf = model(x, teacher_forcing=teacher)
    print(f"  输出 shape: {y_tf.shape}")

    print(f"\n{'='*60}")
    print(f"✅ TrajGRU 模型测试通过!")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_trajgru()
