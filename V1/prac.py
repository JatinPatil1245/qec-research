import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)
qc.h(0)

# Generate the drawing
qc.draw('mpl')

# Force the image to display in a new window
plt.show()