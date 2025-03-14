# 再撑一天也是赚的！
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import math
fontsize_title = 26
fontsize_xlabel = 24
fontsize_ylabel = 24
fontsize_ticks = 20
fontsize_legend = 20


# 定义量子态
state = np.array([1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 1 / math.sqrt(8), 0, 0,
              0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 0, 1 / math.sqrt(8), 1 / math.sqrt(8)])

# 定义量子线路
n_qubits = 4
dev = qml.device("default.mixed", wires=n_qubits)  # 使用支持混合态的设备

@qml.qnode(dev)
def encode_state():
    qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
    return qml.state()  # 直接返回量子态的向量形式

# 初始化量子态
original_state = encode_state()

# 确保 original_state 是一维向量（而不是矩阵）
original_state = original_state.flatten()  # 扁平化为 (16,)

# 定义噪声强度范围
noise_levels = np.linspace(0, 1, 20)

# 用来存储每种噪声下的保真度
phase_shift_fidelity = []
amplitude_damping_fidelity = []
depolarizing_fidelity = []

# Phase Shift噪声：确保每个量子比特都添加相同的噪声强度
for level in noise_levels:
    @qml.qnode(dev)
    def phase_shift_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  # 逐个量子比特应用相位移
            qml.PhaseShift(level, wires=i)
        return qml.state()

    noisy_state = phase_shift_noise()
    noisy_state = noisy_state.flatten()  # 扁平化为 (16,)
    phase_shift_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# Amplitude Damping噪声：确保每个量子比特都添加相同的噪声强度
for level in noise_levels:
    @qml.qnode(dev)
    def amplitude_damping_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  # 逐个量子比特应用幅度衰减噪声
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()

    noisy_state = amplitude_damping_noise()
    noisy_state = noisy_state.flatten()  # 扁平化为 (16,)
    amplitude_damping_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# Depolarizing Channel噪声：确保每个量子比特都添加相同的噪声强度
for level in noise_levels:
    @qml.qnode(dev)
    def depolarizing_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  # 逐个量子比特应用去极化噪声
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()

    noisy_state = depolarizing_noise()
    noisy_state = noisy_state.flatten()  # 扁平化为 (16,)
    depolarizing_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, phase_shift_fidelity, label="Phase Shift", marker='o', linestyle='-', color='b')
plt.plot(noise_levels, amplitude_damping_fidelity, label="Amplitude Damping ", marker='x', linestyle='-', color='r')
plt.plot(noise_levels, depolarizing_fidelity, label="Depolarizing Channel", marker='s', linestyle='-', color='g')

# 设置标题和标签
plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Noise on all qubit", fontsize=fontsize_title)

# 设置横纵轴刻度的字体大小
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)

# 添加背景刻度线
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 显示图例
plt.legend(fontsize=fontsize_legend)

# 保存图像
title = "Noise_all"
plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')

# 展示图形
plt.show()



# 再撑一天也是赚的！
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import math
fontsize_title = 26
fontsize_xlabel = 24
fontsize_ylabel = 24
fontsize_ticks = 20
fontsize_legend = 20


# 定义量子态
state = np.array([1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 1 / math.sqrt(8), 0, 0,
              0, 1 / math.sqrt(8), 0, 1 / math.sqrt(8), 0, 0, 1 / math.sqrt(8), 1 / math.sqrt(8)])

# 定义量子线路
n_qubits = 4
dev = qml.device("default.mixed", wires=n_qubits)  # 使用支持混合态的设备

@qml.qnode(dev)
def encode_state():
    qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
    return qml.state()  # 直接返回量子态的向量形式

# 初始化量子态
original_state = encode_state()

# 确保 original_state 是一维向量（而不是矩阵）
original_state = original_state.flatten()  # 扁平化为 (16,)

# 定义噪声强度范围
noise_levels = np.linspace(0, 1, 20)

# 用来存储每种噪声下的保真度
phase_shift_fidelity = []
amplitude_damping_fidelity = []
depolarizing_fidelity = []

# Phase Shift噪声：确保每个量子比特都添加相同的噪声强度
for level in noise_levels:
    @qml.qnode(dev)
    def phase_shift_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  # 逐个量子比特应用相位移
            qml.PhaseShift(level, wires=i)
        return qml.state()

    noisy_state = phase_shift_noise()
    noisy_state = noisy_state.flatten()  # 扁平化为 (16,)
    phase_shift_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# Amplitude Damping噪声：确保每个量子比特都添加相同的噪声强度
for level in noise_levels:
    @qml.qnode(dev)
    def amplitude_damping_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  # 逐个量子比特应用幅度衰减噪声
            qml.AmplitudeDamping(level, wires=i)
        return qml.state()

    noisy_state = amplitude_damping_noise()
    noisy_state = noisy_state.flatten()  # 扁平化为 (16,)
    amplitude_damping_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# Depolarizing Channel噪声：确保每个量子比特都添加相同的噪声强度
for level in noise_levels:
    @qml.qnode(dev)
    def depolarizing_noise():
        qml.AmplitudeEmbedding(state, wires=range(n_qubits), normalize=True)
        for i in range(n_qubits):  # 逐个量子比特应用去极化噪声
            qml.DepolarizingChannel(level, wires=i)
        return qml.state()

    noisy_state = depolarizing_noise()
    noisy_state = noisy_state.flatten()  # 扁平化为 (16,)
    depolarizing_fidelity.append(np.abs(np.dot(np.conj(original_state), noisy_state)) ** 2)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, phase_shift_fidelity, label="Phase Shift", marker='o', linestyle='-', color='b')
plt.plot(noise_levels, amplitude_damping_fidelity, label="Amplitude Damping ", marker='x', linestyle='-', color='r')
plt.plot(noise_levels, depolarizing_fidelity, label="Depolarizing Channel", marker='s', linestyle='-', color='g')

# 设置标题和标签
plt.xlabel("Noise Strength", fontsize=fontsize_xlabel)
plt.ylabel("Fidelity", fontsize=fontsize_ylabel)
plt.title("Noise on all qubit", fontsize=fontsize_title)

# 设置横纵轴刻度的字体大小
plt.xticks(fontsize=fontsize_ticks)
plt.yticks(fontsize=fontsize_ticks)

# 添加背景刻度线
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

# 显示图例
plt.legend(fontsize=fontsize_legend)

# 保存图像
title = "Noise_all"
plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')

# 展示图形
plt.show()