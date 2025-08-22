import numpy as np

def pack10(data : np.ndarray) -> np.ndarray:
    out = np.zeros((data.shape[0], int(data.shape[1]*(1.25))), dtype=np.uint8)
    out[:, ::5] = data[:, ::4] >> 2
    out[:, 1::5] = ((data[:, ::4] & 0b0000000000000011) << 6)
    out[:, 1::5] += data[:, 1::4] >> 4
    out[:, 2::5] = ((data[:, 1::4] & 0b0000000000001111) << 4)
    out[:, 2::5] += data[:, 2::4] >> 6
    out[:, 3::5] = ((data[:, 2::4] & 0b0000000000111111) << 2)
    out[:, 3::5] += data[:, 3::4] >> 8
    out[:, 4::5] = data[:, 3::4] & 0b0000000011111111
    return out

def pack12(data : np.ndarray) -> np.ndarray:
    out = np.zeros((data.shape[0], int(data.shape[1]*(1.5))), dtype=np.uint8)
    out[:, ::3] = data[:, ::2] >> 4
    out[:, 1::3] = ((data[:, ::2] & 0b0000000000001111) << 4)
    out[:, 1::3] += data[:, 1::2] >> 8
    out[:, 2::3] = data[:, 1::2] & 0b0000001111111111
    return out

def pack14(data : np.ndarray) -> np.ndarray:
    out = np.zeros((data.shape[0], int(data.shape[1]*(1.75))), dtype=np.uint8)

    # 14-bit packing: 4 pixels (56 bits) pack into 7 bytes (56 bits)
    # This gives us the 1.75 compression ratio (7/4 = 1.75)

    # For every group of 4 input pixels, we produce 7 output bytes
    for i in range(0, data.shape[1], 4):
        if i + 3 < data.shape[1]:  # We have a complete group of 4 pixels
            out_base = (i // 4) * 7
            if out_base + 6 < out.shape[1]:  # Make sure we don't exceed output bounds
                # Pixel 0: bits 13-6 go to byte 0, bits 5-0 go to byte 1 (shifted left 2)
                out[:, out_base] = data[:, i] >> 6
                out[:, out_base + 1] = (data[:, i] & 0x3F) << 2

                # Pixel 1: bits 13-12 go to byte 1 (bits 1-0), bits 11-4 go to byte 2, bits 3-0 go to byte 3 (shifted left 4)
                out[:, out_base + 1] |= data[:, i + 1] >> 12
                out[:, out_base + 2] = (data[:, i + 1] >> 4) & 0xFF
                out[:, out_base + 3] = (data[:, i + 1] & 0x0F) << 4

                # Pixel 2: bits 13-10 go to byte 3 (bits 3-0), bits 9-2 go to byte 4, bits 1-0 go to byte 5 (shifted left 6)
                out[:, out_base + 3] |= data[:, i + 2] >> 10
                out[:, out_base + 4] = (data[:, i + 2] >> 2) & 0xFF
                out[:, out_base + 5] = (data[:, i + 2] & 0x03) << 6

                # Pixel 3: bits 13-8 go to byte 5 (bits 5-0), bits 7-0 go to byte 6
                out[:, out_base + 5] |= data[:, i + 3] >> 8
                out[:, out_base + 6] = data[:, i + 3] & 0xFF
        else:
            # Handle remaining pixels (less than 4)
            remaining = data.shape[1] - i
            out_base = (i // 4) * 7
            for j in range(remaining):
                if out_base + j < out.shape[1]:
                    # For incomplete groups, just store the lower 8 bits
                    out[:, out_base + j] = data[:, i + j] & 0xFF

    return out