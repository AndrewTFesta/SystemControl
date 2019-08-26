"""

"""
import numpy as np
import timeit
import struct

from SystemControl.DataConversion import conv24bitsToInt, decompressDeltas19Bit, conv8bitToInt8, decompressDeltas18Bit, \
    scale_fac_uVolts_per_count


class OpenBCISample(object):
    """
    Object encapsulating a single sample from the OpenBCI board.
    """

    def __init__(self, packet_id, channel_data, aux_data, imp_data):
        self.id = packet_id
        self.channel_data = channel_data
        self.aux_data = aux_data
        self.imp_data = imp_data


class GanglionDelegate(DefaultDelegate):
    """
    Called by bluepy (handling BLE connection) when new data arrive, parses samples.
    """

    def __init__(self, scaling_output=True):
        DefaultDelegate.__init__(self)
        # holds samples until OpenBCIBoard claims them
        self.samples = []
        # detect gaps between packets
        self.last_id = -1
        self.packets_dropped = 0
        # save uncompressed data to compute deltas
        self.lastChannelData = [0, 0, 0, 0]
        # 18bit data got here and then accelerometer with it
        self.lastAcceleromoter = [0, 0, 0]
        # when the board is manually set in the right mode (z to start, Z to stop)
        # impedance will be measured. 4 channels + ref
        self.lastImpedance = [0, 0, 0, 0, 0]
        self.scaling_output = scaling_output
        # handling incoming ASCII messages
        self.receiving_ASCII = False
        self.time_last_ASCII = timeit.default_timer()

    def handleNotification(self, cHandle, data):
        if len(data) < 1:
            print('Warning: a packet should at least hold one byte...')
            return
        self.parse(data)

    """
      PARSER:
      Parses incoming data packet into OpenBCISample -- see docs. 
      Will call the corresponding parse* function depending on the format of the packet.
    """

    def parse(self, packet):
        # bluepy returns INT with python3 and STR with python2
        if type(packet) is str:
            # convert a list of strings in bytes
            unpac = struct.unpack(str(len(packet)) + 'B', "".join(packet))
        else:
            unpac = packet

        start_byte = unpac[0]

        # Give the informative part of the packet to proper handler
        # split between ID and data bytes
        # Raw uncompressed
        if start_byte == 0:
            self.receiving_ASCII = False
            self.parseRaw(start_byte, unpac[1:])
        # 18-bit compression with Accelerometer
        elif start_byte >= 1 and start_byte <= 100:
            self.receiving_ASCII = False
            self.parse18bit(start_byte, unpac[1:])
        # 19-bit compression without Accelerometer
        elif start_byte >= 101 and start_byte <= 200:
            self.receiving_ASCII = False
            self.parse19bit(start_byte - 100, unpac[1:])
        # Impedance Channel
        elif start_byte >= 201 and start_byte <= 205:
            self.receiving_ASCII = False
            self.parseImpedance(start_byte, packet[1:])
        # Part of ASCII -- TODO: better formatting of incoming ASCII
        elif start_byte == 206:
            print("%\t" + str(packet[1:]))
            self.receiving_ASCII = True
            self.time_last_ASCII = timeit.default_timer()

            # End of ASCII message
        elif start_byte == 207:
            print("%\t" + str(packet[1:]))
            print("$$$")
            self.receiving_ASCII = False
        else:
            print("Warning: unknown type of packet: " + str(start_byte))

    def parseRaw(self, packet_id, packet):
        """ Dealing with "Raw uncompressed" """
        if len(packet) != 19:
            print('Wrong size, for raw data' +
                  str(len(packet)) + ' instead of 19 bytes')
            return

        chan_data = []
        # 4 channels of 24bits, take values one by one
        for i in range(0, 12, 3):
            chan_data.append(conv24bitsToInt(packet[i:i + 3]))
        # save uncompressed raw channel for future use and append whole sample
        self.pushSample(packet_id, chan_data,
                        self.lastAcceleromoter, self.lastImpedance)
        self.lastChannelData = chan_data
        self.updatePacketsCount(packet_id)

    def parse19bit(self, packet_id, packet):
        """ Dealing with "19-bit compression without Accelerometer" """
        if len(packet) != 19:
            print('Wrong size, for 19-bit compression data' +
                  str(len(packet)) + ' instead of 19 bytes')
            return

        # should get 2 by 4 arrays of uncompressed data
        deltas = decompressDeltas19Bit(packet)
        # the sample_id will be shifted
        delta_id = 1
        for delta in deltas:
            # convert from packet to sample id
            sample_id = (packet_id - 1) * 2 + delta_id
            # 19bit packets hold deltas between two samples
            # TODO: use more broadly numpy
            full_data = list(np.array(self.lastChannelData) - np.array(delta))
            # NB: aux data updated only in 18bit mode, send values here only to be consistent
            self.pushSample(sample_id, full_data,
                            self.lastAcceleromoter, self.lastImpedance)
            self.lastChannelData = full_data
            delta_id += 1
        self.updatePacketsCount(packet_id)

    def parse18bit(self, packet_id, packet):
        """ Dealing with "18-bit compression without Accelerometer" """
        if len(packet) != 19:
            print('Wrong size, for 18-bit compression data' +
                  str(len(packet)) + ' instead of 19 bytes')
            return

        # accelerometer X
        if packet_id % 10 == 1:
            self.lastAcceleromoter[0] = conv8bitToInt8(packet[18])
        # accelerometer Y
        elif packet_id % 10 == 2:
            self.lastAcceleromoter[1] = conv8bitToInt8(packet[18])
        # accelerometer Z
        elif packet_id % 10 == 3:
            self.lastAcceleromoter[2] = conv8bitToInt8(packet[18])

        # deltas: should get 2 by 4 arrays of uncompressed data
        deltas = decompressDeltas18Bit(packet[:-1])
        # the sample_id will be shifted
        delta_id = 1
        for delta in deltas:
            # convert from packet to sample id
            sample_id = (packet_id - 1) * 2 + delta_id
            # 19bit packets hold deltas between two samples
            # TODO: use more broadly numpy
            full_data = list(np.array(self.lastChannelData) - np.array(delta))
            self.pushSample(sample_id, full_data,
                            self.lastAcceleromoter, self.lastImpedance)
            self.lastChannelData = full_data
            delta_id += 1
        self.updatePacketsCount(packet_id)

    def parseImpedance(self, packet_id, packet):
        """ Dealing with impedance data. packet: ASCII data.
        NB: will take few packet (seconds) to fill
        """
        if packet[-2:] != b"Z\n":
            print('Wrong format for impedance check, should be ASCII ending with "Z\\n"')

        # convert from ASCII to actual value
        imp_value = int(packet[:-2]) / 2
        # from 201 to 205 codes to the right array size
        self.lastImpedance[packet_id - 201] = imp_value
        self.pushSample(packet_id - 200, self.lastChannelData,
                        self.lastAcceleromoter, self.lastImpedance)

    def pushSample(self, sample_id, chan_data, aux_data, imp_data):
        """ Add a sample to inner stack, setting ID and dealing with scaling if necessary. """
        if self.scaling_output:
            chan_data = list(np.array(chan_data) * scale_fac_uVolts_per_count)
            aux_data = list(np.array(aux_data) * scale_fac_accel_G_per_count)
        sample = OpenBCISample(sample_id, chan_data, aux_data, imp_data)
        self.samples.append(sample)

    def updatePacketsCount(self, packet_id):
        """Update last packet ID and dropped packets"""
        if self.last_id == -1:
            self.last_id = packet_id
            self.packets_dropped = 0
            return
        # ID loops every 101 packets
        if packet_id > self.last_id:
            self.packets_dropped = packet_id - self.last_id - 1
        else:
            self.packets_dropped = packet_id + 101 - self.last_id - 1
        self.last_id = packet_id
        if self.packets_dropped > 0:
            print("Warning: dropped " + str(self.packets_dropped) + " packets.")

    def getSamples(self):
        """ Retrieve and remove from buffer last samples. """
        unstack_samples = self.samples
        self.samples = []
        return unstack_samples

    def getMaxPacketsDropped(self):
        """ While processing last samples, how many packets were dropped?"""
        # TODO: return max value of the last samples array?
        return self.packets_dropped
