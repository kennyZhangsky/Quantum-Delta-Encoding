import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import math
fontsize_title = 26
fontsize_xlabel = 24
fontsize_ylabel = 24
fontsize_ticks = 20
fontsize_legend = 20


# Theoretical QDE state defined. Detail quantum circuit construction refer to QDE_case.py

state = np.array([1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 1 / math.sqrt(8), 0, 0,
              0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 0, 1 / math.sqrt(8), 1 / math.sqrt(8)])

n_qubits = 4
dev = qml.device("default.mixed", wires=n_qubits)  

@qml.qnode(dev)
def encode_state():
    qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
    return qml.state()  

# int
original_state = encode_state()

# data flatten
original_state = original_state.flatten()  

# noise range define
noise_levels = np.linspace(0, 1, 20)

phase_shift_fidelity = []
amplitude_damping_fidelity = []
depolarizing_fidelity = []

# Phase Shift
for level in noise_levels:
    @qml.qnode(dev)
    def phase_shift_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  
            qml.PhaseShift(level, wires=i)
        return qml.state()

    noisy_state = phase_shift_noise()
    noisy_state = noisy_state.flatten()  
    phase_shift_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# Amplitude Damping
for level in noise_levels:
    @qml.qnode(dev)
    def amplitude_damping_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()

    noisy_state = amplitude_damping_noise()
    noisy_state = noisy_state.flatten()  
    amplitude_damping_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# Depolarizing Channel
for level in noise_levels:
    @qml.qnode(dev)
    def depolarizing_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()

    noisy_state = depolarizing_noise()
    noisy_state = noisy_state.flatten()  
    depolarizing_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# pics saving
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, phase_shift_fidelity, label="Phase Shift", marker='o', linestyle='-', color='b')
plt.plot(noise_levels, amplitude_damping_fidelity, label="Amplitude Damping ", marker='x', linestyle='-', color='r')
plt.plot(noise_levels, depolarizing_fidelity, label="Depolarizing Channel", marker='s', linestyle='-', color='g')

plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Noise on all qubit", fontsize=fontsize_title)

plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

plt.legend(fontsize=fontsize_legend)

title = "Noise_all"
plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')

plt.show()
