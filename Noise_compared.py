import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import math

# noise comparison of QDE and BRQI and CE-NGQR, denoted as state, state2 and state3. detailed circuit construction please refer to their references.

# font size setting
fontsize_title = 26
fontsize_xlabel = 24
fontsize_ylabel = 24
fontsize_ticks = 20
fontsize_legend = 20

control_patterns = [
        "001100", "001010", "000110", "001110", "000001", "001001", "000101",
        "100000", "101100", "101010", "100110", "101110", "100001", "101001", "100101",
        "011000", "011100", "011010", "010110", "011110", "010001", "011001", "010101",
        "110000", "111000", "111100", "111010", "110110", "111110", "110001", "111001", "110101"
    ]

# qde state init
state = np.array([1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 1 / math.sqrt(8), 0, 0,
                  0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 0, 1 / math.sqrt(8), 1 / math.sqrt(8)])

# 
n_qubits1 = 4   
n_qubits2 = 8   
n_qubits3 = 8
# 
dev1 = qml.device("default.mixed", wires=n_qubits1)
dev2 = qml.device("default.mixed", wires=n_qubits2)
dev3 = qml.device("default.mixed", wires=n_qubits3)
#  state1 circuit
@qml.qnode(dev1)
def encode_state():
    qml.AmplitudeEmbedding(state, wires=range(n_qubits1), normalize=True)
    return qml.state()

# state2 and 3 circuit

@qml.qnode(dev2)
def encode_state2():
    for i in range(1, 7):
        qml.Hadamard(wires=i)
    for target in [0, 7]:  
        for pattern in control_patterns:
            control_wires = [1, 2, 3, 4, 5, 6]
            control_values = [int(bit) for bit in pattern]
            qml.MultiControlledX(control_wires=control_wires, wires=[target], control_values=control_values)
    return qml.state()


@qml.qnode(dev3)
def encode_state3():
    for i in range(1, 7):
        qml.Hadamard(wires=i)

    for control_values in control_patterns:
        controls = [int(c) for c in control_values]  
        qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=0, control_values=controls)

    qml.RY(np.pi / 2, wires=7)
    return qml.state()

# flatten
original_state = encode_state().flatten()
original_state2 = encode_state2().flatten()
original_state3 = encode_state3().flatten()

# noise range
noise_levels = np.linspace(0, 1, 20)

# 
phase_shift_fidelity_state1 = []
phase_shift_fidelity_state2 = []
phase_shift_fidelity_state3 = []

amplitude_damping_fidelity_state1 = []
amplitude_damping_fidelity_state2 = []
amplitude_damping_fidelity_state3 = []

depolarizing_fidelity_state1 = []
depolarizing_fidelity_state2 = []
depolarizing_fidelity_state3 = []

# ======================
# Phase Shift 
# ======================
for level in noise_levels:
    #  state1 (dev1)
    @qml.qnode(dev1)
    def phase_shift_noise_state1():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits1), normalize=True)
        for i in range(n_qubits1):
            qml.PhaseShift(level, wires=i)
        return qml.state()
    noisy_state1 = phase_shift_noise_state1().flatten()
    fidelity1 = np.abs(np.dot(np.conj(original_state), noisy_state1)) ** 2
    phase_shift_fidelity_state1.append(fidelity1)

    # 
    @qml.qnode(dev2)
    def phase_shift_noise_state2():
        for i in range(1, 7):
            qml.Hadamard(wires=i)

        for target in [0, 7]:  
            for pattern in control_patterns:
                control_wires = [1, 2, 3, 4, 5, 6]
                control_values = [int(bit) for bit in pattern]
                qml.MultiControlledX(control_wires=control_wires, wires=[target], control_values=control_values)

        for i in range(n_qubits2):
            qml.PhaseShift(level, wires=i)
        return qml.state()
    noisy_state2 = phase_shift_noise_state2().flatten()
    fidelity2 = np.abs(np.dot(np.conj(original_state2), noisy_state2)) ** 2
    phase_shift_fidelity_state2.append(fidelity2)

    #  state3 (dev2) 
    @qml.qnode(dev3)
    def phase_shift_noise_state3():
        for i in range(1, 7):
            qml.Hadamard(wires=i)
        
        for control_values in control_patterns:
            controls = [int(c) for c in control_values]
            qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=0, control_values=controls)
        
        qml.RY(np.pi / 2, wires=7)

        for i in range(n_qubits3):
            qml.PhaseShift(level, wires=i)
        return qml.state()
    noisy_state3 = phase_shift_noise_state3().flatten()
    fidelity3 = np.abs(np.dot(np.conj(original_state3), noisy_state3)) ** 2
    phase_shift_fidelity_state3.append(fidelity3)

# ======================
# Amplitude Damping 
# ======================
for level in noise_levels:
    #  state1 (dev1)
    @qml.qnode(dev1)
    def amplitude_damping_noise_state1():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits1), normalize=True)
        for i in range(n_qubits1):
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()
    noisy_state1 = amplitude_damping_noise_state1().flatten()
    fidelity1 = np.abs(np.dot(np.conj(original_state), noisy_state1)) ** 2
    amplitude_damping_fidelity_state1.append(fidelity1)

    #  state2 (dev2)
    @qml.qnode(dev2)
    def amplitude_damping_noise_state2():
        for i in range(1, 7):
            qml.Hadamard(wires=i)

        for target in [0, 7]:  
            for pattern in control_patterns:
                control_wires = [1, 2, 3, 4, 5, 6]
                control_values = [int(bit) for bit in pattern]
                qml.MultiControlledX(control_wires=control_wires, wires=[target], control_values=control_values)
        for i in range(n_qubits2):
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()
    noisy_state2 = amplitude_damping_noise_state2().flatten()
    fidelity2 = np.abs(np.dot(np.conj(original_state2), noisy_state2)) ** 2
    amplitude_damping_fidelity_state2.append(fidelity2)

    # 针对 state3 (dev2)
    @qml.qnode(dev3)
    def amplitude_damping_noise_state3():
        for i in range(1, 7):
            qml.Hadamard(wires=i)
        
        for control_values in control_patterns:
            controls = [int(c) for c in control_values]
            qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=0, control_values=controls)
        qml.RY(np.pi / 2, wires=[7])
        for i in range(n_qubits3):
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()
    noisy_state3 = amplitude_damping_noise_state3().flatten()
    fidelity3 = np.abs(np.dot(np.conj(original_state3), noisy_state3)) ** 2
    amplitude_damping_fidelity_state3.append(fidelity3)

# ======================
# Depolarizing Channel 
# ======================
#noise range rescale 0.5
noise_levels1 = np.linspace(0, 0.5, 20)
for level in noise_levels1:
    #  state1 (dev1)
    @qml.qnode(dev1)
    def depolarizing_noise_state1():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits1), normalize=True)
        for i in range(n_qubits1):
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()
    noisy_state1 = depolarizing_noise_state1().flatten()
    fidelity1 = np.abs(np.dot(np.conj(original_state), noisy_state1)) ** 2
    depolarizing_fidelity_state1.append(fidelity1)

    #  state2 (dev2)
    @qml.qnode(dev2)
    def depolarizing_noise_state2():
        for i in range(1, 7):
            qml.Hadamard(wires=i)

        for target in [0, 7]:  
            for pattern in control_patterns:
                control_wires = [1, 2, 3, 4, 5, 6]
                control_values = [int(bit) for bit in pattern]
                qml.MultiControlledX(control_wires=control_wires, wires=[target], control_values=control_values)
        for i in range(n_qubits2):
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()
    noisy_state2 = depolarizing_noise_state2().flatten()
    fidelity2 = np.abs(np.dot(np.conj(original_state2), noisy_state2)) ** 2
    depolarizing_fidelity_state2.append(fidelity2)

    #  state3 (dev2)
    @qml.qnode(dev3)
    def depolarizing_noise_state3():
        for i in range(1, 7):
            qml.Hadamard(wires=i)
        
        for control_values in control_patterns:
            controls = [int(c) for c in control_values]
            qml.MultiControlledX(control_wires=[1, 2, 3, 4, 5, 6], wires=0, control_values=controls)
        qml.RY(np.pi / 2, wires=[7])
        for i in range(n_qubits3):
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()
    noisy_state3 = depolarizing_noise_state3().flatten()
    fidelity3 = np.abs(np.dot(np.conj(original_state3), noisy_state3)) ** 2
    depolarizing_fidelity_state3.append(fidelity3)

# ======================
# saving Phase Shift pics
# ======================
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, phase_shift_fidelity_state1, label="QDE", marker='o', linestyle='-', color='#264653')
plt.plot(noise_levels, phase_shift_fidelity_state2, label="PE-NGQR", marker='s', linestyle='-', color='#E66F51')
plt.plot(noise_levels, phase_shift_fidelity_state3, label="MQIR", marker='d', linestyle='-', color='#2A9D8E')
plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Phase Shift on QDE, MQIR and PE-NGQR", fontsize=fontsize_title)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.legend(fontsize=fontsize_legend)
plt.savefig("phase_shift_noise.png", dpi=300, bbox_inches='tight')
plt.show()

# ======================
# saving Amplitude Damping pics
# ======================
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, amplitude_damping_fidelity_state1, label="QDE", marker='o', linestyle='-', color='#264653')
plt.plot(noise_levels, amplitude_damping_fidelity_state2, label="PE-NGQR", marker='s', linestyle='-', color='#E66F51')
plt.plot(noise_levels, amplitude_damping_fidelity_state3, label="MQIR", marker='d', linestyle='-', color='#2A9D8E')
plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Amplitude Damping on QDE, MQIR and PE-NGQR", fontsize=fontsize_title)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.legend(fontsize=fontsize_legend)
plt.savefig("amplitude_damping_noise.png", dpi=300, bbox_inches='tight')
plt.show()

# ======================
# saving Depolarizing Channel pics
# ======================
plt.figure(figsize=(10, 6))
plt.plot(noise_levels1, depolarizing_fidelity_state1, label="QDE", marker='o', linestyle='-', color='#264653')
plt.plot(noise_levels1, depolarizing_fidelity_state2, label="PE-NGQR", marker='s', linestyle='-', color='#E66F51')
plt.plot(noise_levels1, depolarizing_fidelity_state3, label="MQIR", marker='d', linestyle='-', color='#2A9D8E')
plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Depolarizing Channel on QDE, MQIR and PE-NGQR", fontsize=fontsize_title)
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.legend(fontsize=fontsize_legend)
plt.savefig("depolarizing_noise.png", dpi=300, bbox_inches='tight')
plt.show()

