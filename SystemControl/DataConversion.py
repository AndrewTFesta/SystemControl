"""

"""
import struct

SAMPLE_RATE = 200.0  # Hz
scale_fac_uVolts_per_count = 1200 / (8388607.0 * 1.5 * 51.0)
scale_fac_accel_G_per_count = 0.000016


def conv24bitsToInt(unpacked):
    """ Convert 24bit data coded on 3 bytes to a proper integer """
    if len(unpacked) != 3:
        raise ValueError("Input should be 3 bytes long.")

    # FIXME: quick'n dirty, unpack wants strings later on
    literal_read = struct.pack('3B', unpacked[0], unpacked[1], unpacked[2])

    # 3byte int in 2s compliment
    if unpacked[0] > 127:
        pre_fix = bytes(bytearray.fromhex('FF'))
    else:
        pre_fix = bytes(bytearray.fromhex('00'))

    literal_read = pre_fix + literal_read

    # unpack little endian(>) signed integer(i) (makes unpacking platform independent)
    myInt = struct.unpack('>i', literal_read)[0]

    return myInt


def conv19bitToInt32(threeByteBuffer):
    """ Convert 19bit data coded on 3 bytes to a proper integer (LSB bit 1 used as sign). """
    if len(threeByteBuffer) != 3:
        raise ValueError("Input should be 3 bytes long.")

    prefix = 0

    # if LSB is 1, negative number, some hasty unsigned to signed conversion to do
    if threeByteBuffer[2] & 0x01 > 0:
        prefix = 0b1111111111111
        return ((prefix << 19) | (threeByteBuffer[0] << 16) |
                (threeByteBuffer[1] << 8) | threeByteBuffer[2]) | ~0xFFFFFFFF
    else:
        return (prefix << 19) | (threeByteBuffer[0] << 16) | \
               (threeByteBuffer[1] << 8) | threeByteBuffer[2]


def conv18bitToInt32(threeByteBuffer):
    """ Convert 18bit data coded on 3 bytes to a proper integer (LSB bit 1 used as sign) """
    if len(threeByteBuffer) != 3:
        raise ValueError("Input should be 3 bytes long.")

    prefix = 0

    # if LSB is 1, negative number, some hasty unsigned to signed conversion to do
    if threeByteBuffer[2] & 0x01 > 0:
        prefix = 0b11111111111111
        return ((prefix << 18) | (threeByteBuffer[0] << 16) |
                (threeByteBuffer[1] << 8) | threeByteBuffer[2]) | ~0xFFFFFFFF
    else:
        return (prefix << 18) | (threeByteBuffer[0] << 16) | \
               (threeByteBuffer[1] << 8) | threeByteBuffer[2]


def conv8bitToInt8(byte):
    """ Convert one byte to signed value """

    if byte > 127:
        return (256 - byte) * (-1)
    else:
        return byte


def decompressDeltas19Bit(buffer):
    """
    Called to when a compressed packet is received.
    buffer: Just the data portion of the sample. So 19 bytes.
    return {Array} - An array of deltas of shape 2x4
    (2 samples per packet and 4 channels per sample.)
    """
    if len(buffer) != 19:
        raise ValueError("Input should be 19 bytes long.")

    receivedDeltas = [[0, 0, 0, 0], [0, 0, 0, 0]]

    # Sample 1 - Channel 1
    miniBuf = [
        (buffer[0] >> 5),
        ((buffer[0] & 0x1F) << 3 & 0xFF) | (buffer[1] >> 5),
        ((buffer[1] & 0x1F) << 3 & 0xFF) | (buffer[2] >> 5)
    ]

    receivedDeltas[0][0] = conv19bitToInt32(miniBuf)

    # Sample 1 - Channel 2
    miniBuf = [
        (buffer[2] & 0x1F) >> 2,
        (buffer[2] << 6 & 0xFF) | (buffer[3] >> 2),
        (buffer[3] << 6 & 0xFF) | (buffer[4] >> 2)
    ]
    receivedDeltas[0][1] = conv19bitToInt32(miniBuf)

    # Sample 1 - Channel 3
    miniBuf = [
        ((buffer[4] & 0x03) << 1 & 0xFF) | (buffer[5] >> 7),
        ((buffer[5] & 0x7F) << 1 & 0xFF) | (buffer[6] >> 7),
        ((buffer[6] & 0x7F) << 1 & 0xFF) | (buffer[7] >> 7)
    ]
    receivedDeltas[0][2] = conv19bitToInt32(miniBuf)

    # Sample 1 - Channel 4
    miniBuf = [
        ((buffer[7] & 0x7F) >> 4),
        ((buffer[7] & 0x0F) << 4 & 0xFF) | (buffer[8] >> 4),
        ((buffer[8] & 0x0F) << 4 & 0xFF) | (buffer[9] >> 4)
    ]
    receivedDeltas[0][3] = conv19bitToInt32(miniBuf)

    # Sample 2 - Channel 1
    miniBuf = [
        ((buffer[9] & 0x0F) >> 1),
        (buffer[9] << 7 & 0xFF) | (buffer[10] >> 1),
        (buffer[10] << 7 & 0xFF) | (buffer[11] >> 1)
    ]
    receivedDeltas[1][0] = conv19bitToInt32(miniBuf)

    # Sample 2 - Channel 2
    miniBuf = [
        ((buffer[11] & 0x01) << 2 & 0xFF) | (buffer[12] >> 6),
        (buffer[12] << 2 & 0xFF) | (buffer[13] >> 6),
        (buffer[13] << 2 & 0xFF) | (buffer[14] >> 6)
    ]
    receivedDeltas[1][1] = conv19bitToInt32(miniBuf)

    # Sample 2 - Channel 3
    miniBuf = [
        ((buffer[14] & 0x38) >> 3),
        ((buffer[14] & 0x07) << 5 & 0xFF) | ((buffer[15] & 0xF8) >> 3),
        ((buffer[15] & 0x07) << 5 & 0xFF) | ((buffer[16] & 0xF8) >> 3)
    ]
    receivedDeltas[1][2] = conv19bitToInt32(miniBuf)

    # Sample 2 - Channel 4
    miniBuf = [(buffer[16] & 0x07), buffer[17], buffer[18]]
    receivedDeltas[1][3] = conv19bitToInt32(miniBuf)

    return receivedDeltas


def decompressDeltas18Bit(buffer):
    """
    Called to when a compressed packet is received.
    buffer: Just the data portion of the sample. So 19 bytes.
    return {Array} - An array of deltas of shape 2x4
    (2 samples per packet and 4 channels per sample.)
    """
    if len(buffer) != 18:
        raise ValueError("Input should be 18 bytes long.")

    receivedDeltas = [[0, 0, 0, 0], [0, 0, 0, 0]]

    # Sample 1 - Channel 1
    miniBuf = [
        (buffer[0] >> 6),
        ((buffer[0] & 0x3F) << 2 & 0xFF) | (buffer[1] >> 6),
        ((buffer[1] & 0x3F) << 2 & 0xFF) | (buffer[2] >> 6)
    ]
    receivedDeltas[0][0] = conv18bitToInt32(miniBuf)

    # Sample 1 - Channel 2
    miniBuf = [
        (buffer[2] & 0x3F) >> 4,
        (buffer[2] << 4 & 0xFF) | (buffer[3] >> 4),
        (buffer[3] << 4 & 0xFF) | (buffer[4] >> 4)
    ]
    receivedDeltas[0][1] = conv18bitToInt32(miniBuf)

    # Sample 1 - Channel 3
    miniBuf = [
        (buffer[4] & 0x0F) >> 2,
        (buffer[4] << 6 & 0xFF) | (buffer[5] >> 2),
        (buffer[5] << 6 & 0xFF) | (buffer[6] >> 2)
    ]
    receivedDeltas[0][2] = conv18bitToInt32(miniBuf)

    # Sample 1 - Channel 4
    miniBuf = [
        (buffer[6] & 0x03),
        buffer[7],
        buffer[8]
    ]
    receivedDeltas[0][3] = conv18bitToInt32(miniBuf)

    # Sample 2 - Channel 1
    miniBuf = [
        (buffer[9] >> 6),
        ((buffer[9] & 0x3F) << 2 & 0xFF) | (buffer[10] >> 6),
        ((buffer[10] & 0x3F) << 2 & 0xFF) | (buffer[11] >> 6)
    ]
    receivedDeltas[1][0] = conv18bitToInt32(miniBuf)

    # Sample 2 - Channel 2
    miniBuf = [
        (buffer[11] & 0x3F) >> 4,
        (buffer[11] << 4 & 0xFF) | (buffer[12] >> 4),
        (buffer[12] << 4 & 0xFF) | (buffer[13] >> 4)
    ]
    receivedDeltas[1][1] = conv18bitToInt32(miniBuf)

    # Sample 2 - Channel 3
    miniBuf = [
        (buffer[13] & 0x0F) >> 2,
        (buffer[13] << 6 & 0xFF) | (buffer[14] >> 2),
        (buffer[14] << 6 & 0xFF) | (buffer[15] >> 2)
    ]
    receivedDeltas[1][2] = conv18bitToInt32(miniBuf)

    # Sample 2 - Channel 4
    miniBuf = [
        (buffer[15] & 0x03),
        buffer[16],
        buffer[17]
    ]
    receivedDeltas[1][3] = conv18bitToInt32(miniBuf)

    return receivedDeltas
