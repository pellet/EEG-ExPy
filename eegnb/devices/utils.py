import numpy as np
import socket
import sys
import platform
import serial

from brainflow.board_shim import BoardShim, BoardIds


# Default channel names for the various EEG devices.
EEG_CHANNELS = {
    "muse2016": ["TP9", "AF7", "AF8", "TP10", "Right AUX"],
    "muse2": ["TP9", "AF7", "AF8", "TP10", "Right AUX"],
    "museS": ["TP9", "AF7", "AF8", "TP10", "Right AUX"],
    "muse2016_bfn": BoardShim.get_eeg_names(BoardIds.MUSE_2016_BOARD.value),
    "muse2016_bfb": BoardShim.get_eeg_names(BoardIds.MUSE_2016_BLED_BOARD.value),
    "muse2_bfn": BoardShim.get_eeg_names(BoardIds.MUSE_2_BOARD.value),
    "muse2_bfb": BoardShim.get_eeg_names(BoardIds.MUSE_2_BLED_BOARD.value),
    "museS_bfn": BoardShim.get_eeg_names(BoardIds.MUSE_S_BOARD.value),
    "museS_bfb": BoardShim.get_eeg_names(BoardIds.MUSE_S_BLED_BOARD.value),
    "ganglion": ["fp1", "fp2", "tp7", "tp8"],
    "cyton": BoardShim.get_eeg_names(BoardIds.CYTON_BOARD.value),
    "cyton_daisy": BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value),
    "brainbit": BoardShim.get_eeg_names(BoardIds.BRAINBIT_BOARD.value),
    "unicorn": BoardShim.get_eeg_names(BoardIds.UNICORN_BOARD.value),
    "synthetic": BoardShim.get_eeg_names(BoardIds.SYNTHETIC_BOARD.value),
    "notion1": BoardShim.get_eeg_names(BoardIds.NOTION_1_BOARD.value),
    "notion2": BoardShim.get_eeg_names(BoardIds.NOTION_2_BOARD.value),
    "crown": BoardShim.get_eeg_names(BoardIds.CROWN_BOARD.value),
    "freeeeg32": [f"eeg_{i}" for i in range(0, 32)],
    "kernelflow": [],
    "biosemi": [],
    "nirsport2": [],
}

BRAINFLOW_CHANNELS = {
    "ganglion": [],
    "cyton": EEG_CHANNELS["cyton"] + ["accel_0", "accel_1", "accel_2"],
    "cyton_daisy": EEG_CHANNELS["cyton_daisy"] + ["accel_0", "accel_1", "accel_2"],
    "synthetic": EEG_CHANNELS["synthetic"],
}

EEG_INDICES = {
    "muse2016": [1, 2, 3, 4],
    "muse2": [1, 2, 3, 4],
    "museS": [1, 2, 3, 4],
    "muse2016_bfn": BoardShim.get_eeg_channels(BoardIds.MUSE_2016_BOARD.value),
    "muse2016_bfb": BoardShim.get_eeg_channels(BoardIds.MUSE_2016_BLED_BOARD.value),
    "muse2_bfn": BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value),
    "muse2_bfb": BoardShim.get_eeg_channels(BoardIds.MUSE_2_BLED_BOARD.value),
    "museS_bfn": BoardShim.get_eeg_channels(BoardIds.MUSE_S_BOARD.value),
    "museS_bfb": BoardShim.get_eeg_channels(BoardIds.MUSE_S_BLED_BOARD.value),
    "ganglion": BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD.value),
    "cyton": BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value),
    "cyton_daisy": BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value),
    "brainbit": BoardShim.get_eeg_channels(BoardIds.BRAINBIT_BOARD.value),
    "unicorn": BoardShim.get_eeg_channels(BoardIds.UNICORN_BOARD.value),
    "synthetic": BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value),
    "notion1": BoardShim.get_eeg_channels(BoardIds.NOTION_1_BOARD.value),
    "notion2": BoardShim.get_eeg_channels(BoardIds.NOTION_2_BOARD.value),
    "crown": BoardShim.get_eeg_channels(BoardIds.CROWN_BOARD.value),
    "freeeeg32": BoardShim.get_eeg_channels(BoardIds.FREEEEG32_BOARD.value),
    "kernelflow": [],
    "biosemi": [],
    "nirsport2": [],
    }

SAMPLE_FREQS = {
    "muse2016": 256,
    "muse2": 256,
    "museS": 256,
    "muse2016_bfn": BoardShim.get_sampling_rate(BoardIds.MUSE_2016_BOARD.value),
    "muse2016_bfb": BoardShim.get_sampling_rate(BoardIds.MUSE_2016_BLED_BOARD.value),
    "muse2_bfn": BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value),
    "muse2_bfb": BoardShim.get_sampling_rate(BoardIds.MUSE_2_BLED_BOARD.value),
    "museS_bfn": BoardShim.get_sampling_rate(BoardIds.MUSE_S_BOARD.value),
    "museS_bfb": BoardShim.get_sampling_rate(BoardIds.MUSE_S_BLED_BOARD.value),
    "ganglion": BoardShim.get_sampling_rate(BoardIds.GANGLION_BOARD.value),
    "cyton": BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value),
    "cyton_daisy": BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value),
    "brainbit": BoardShim.get_sampling_rate(BoardIds.BRAINBIT_BOARD.value),
    "unicorn": BoardShim.get_sampling_rate(BoardIds.UNICORN_BOARD.value),
    "synthetic": BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value),
    "notion1": BoardShim.get_sampling_rate(BoardIds.NOTION_1_BOARD.value),
    "notion2": BoardShim.get_sampling_rate(BoardIds.NOTION_2_BOARD.value),
    "crown": BoardShim.get_sampling_rate(BoardIds.CROWN_BOARD.value),
    "freeeeg32": BoardShim.get_sampling_rate(BoardIds.FREEEEG32_BOARD.value),
    "kernelflow": [],
    "biosemi": [],
    "nirsport2": [],
    }



# ---------------------------------------------------------------------------
# Cyton board channel configuration presets
# ---------------------------------------------------------------------------
# Each channel command has the format:  x N P G I B S1 S2 X
#   N  = channel number (1-8)
#   P  = power (0=ON, 1=OFF)
#   G  = gain  (0=1×, 1=2×, 2=4×, 3=6×, 4=8×, 5=12×, 6=24×)
#   I  = input type (0=normal EEG, 1=shorted, ...)
#   B  = include in BIAS derivation (1=yes)
#   S2 = SRB2 connection (1=connected)
#   S1 = SRB1 connection (0=disconnected)
#
# Build a config string by joining per-channel strings — applied with
# EEG(device='cyton', config=CYTON_CONFIG_GAIN_4X).

def _cyton_ch_config(gain_code: int, n_channels: int = 8) -> str:
    """Build a Cyton channel-settings string for all channels.

    Args:
        gain_code: ADS1299 gain code (0=1×, 1=2×, 2=4×, 3=6×, 4=8×, 5=12×, 6=24×).
        n_channels: Number of channels to configure (default 8 for standard Cyton).

    Returns:
        Config string ready to pass to ``EEG(config=...)``.
    """
    return "".join(f"x{ch}0{gain_code}0110X" for ch in range(1, n_channels + 1))

# Standard gain presets — normal EEG input, bias enabled, SRB2 on, SRB1 off.
CYTON_CONFIG_GAIN_1X  = _cyton_ch_config(0)   # 1× (for strong signals / testing)
CYTON_CONFIG_GAIN_2X  = _cyton_ch_config(1)   # 2×
CYTON_CONFIG_GAIN_4X  = _cyton_ch_config(2)   # 4×  - for Thinkpulse electrodes
CYTON_CONFIG_GAIN_6X  = _cyton_ch_config(3)   # 6×
CYTON_CONFIG_GAIN_8X  = _cyton_ch_config(4)   # 8×
CYTON_CONFIG_GAIN_12X = _cyton_ch_config(5)   # 12× — good general-purpose EEG config
CYTON_CONFIG_GAIN_24X = _cyton_ch_config(6)   # 24× — default gain


def create_stim_array(timestamps, markers):
    """Creates a stim array which is the lenmgth of the EEG data where the stimuli are lined up
    with their corresponding EEG sample.
    Parameters:
        timestamps (array of floats): Timestamps from the EEG data.
        markers (array of ints): Markers and their associated timestamps.
    """
    # marker_max = np.max(markers)
    num_samples = len(timestamps)
    stim_array = np.zeros((num_samples, 1))
    for marker in markers:
        stim_idx = np.where(timestamps == marker[1])
        stim_array[stim_idx] = marker[0]

    return stim_array


def get_openbci_usb():
    print("\nGetting a list of available serial ports...")
    port_list = serial_ports()
    i = 0
    for port in port_list:
        print(f"[{i}] {port}")
        i += 1
    port_number = input("Select Port(number): ")
    if port_number == "":
        return port_list[int(input("This field is required. Select Port(number): "))]
    else:
        return str(port_list[int(port_number)]).split(" - ")[0]


def serial_ports():
    return serial.tools.list_ports.comports()


def get_ftdi_latency_ms(com_port: str):
    """Read the FTDI VCP LatencyTimer (ms) for a COM port from the Windows registry.

    Returns the latency in ms, or None on non-Windows platforms or if the port
    is not found under HKLM\\SYSTEM\\CurrentControlSet\\Enum\\FTDIBUS.
    """
    if sys.platform != 'win32':
        return None
    import winreg
    ftdi_root = r"SYSTEM\CurrentControlSet\Enum\FTDIBUS"
    target = com_port.upper()
    try:
        root_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, ftdi_root)
    except OSError:
        return None
    try:
        for i in range(1024):
            try:
                device_name = winreg.EnumKey(root_key, i)
            except OSError:
                break
            device_path = f"{ftdi_root}\\{device_name}"
            try:
                device_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, device_path)
            except OSError:
                continue
            try:
                for j in range(64):
                    try:
                        instance = winreg.EnumKey(device_key, j)
                    except OSError:
                        break
                    params_path = f"{device_path}\\{instance}\\Device Parameters"
                    try:
                        params_key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, params_path)
                    except OSError:
                        continue
                    try:
                        port_name, _ = winreg.QueryValueEx(params_key, "PortName")
                        if str(port_name).upper() == target:
                            latency, _ = winreg.QueryValueEx(params_key, "LatencyTimer")
                            return int(latency)
                    except OSError:
                        pass
                    finally:
                        params_key.Close()
            finally:
                device_key.Close()
    finally:
        root_key.Close()
    return None


def assert_ftdi_latency_1ms(com_port: str) -> None:
    """Assert the FTDI LatencyTimer for a COM port is 1 ms (Windows only).

    The OpenBCI Cyton dongle ships with the Windows default of 16 ms, which
    adds ~15 ms of USB buffering jitter to every marker push and corrupts
    stimulus/EEG alignment for VEP-class experiments. No-op on non-Windows.

    Fix in: Device Manager -> Ports -> USB Serial Port -> Properties ->
    Port Settings -> Advanced -> Latency Timer (ms) = 1.
    """
    if sys.platform != 'win32':
        return
    latency = get_ftdi_latency_ms(com_port)
    assert latency is not None, (
        f"Could not read FTDI LatencyTimer for {com_port}. "
        f"Verify it is an FTDI device in Device Manager."
    )
    assert latency == 1, (
        f"FTDI LatencyTimer for {com_port} is {latency} ms; required 1 ms. "
        f"Device Manager -> Ports -> USB Serial Port ({com_port}) -> "
        f"Properties -> Port Settings -> Advanced -> Latency Timer (ms) = 1."
    )
