import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# 1. RUTA AL ARCHIVO .MAT
file_path = r"D:\\Violeta\\ANALISIS\\WT_90\\2025_07_16_0000\\rippleCSDs.mat"

# Cargar el archivo .mat
mat_data = scipy.io.loadmat(file_path)

# 2. EXTRAER LA VARIABLE DE CSD
csd_data = mat_data['CSDs']  # Matriz de CSD (canales x tiempo)

# 3. CONFIGURAR LA ESCALA DE COLOR 
# Calculamos el valor máximo absoluto para que el 0 quede exactamente en el centro (color blanco)
vmax = np.max(np.abs(csd_data))
vmin = -vmax

# 4. GRAFICAR EL CSD
fig, ax = plt.subplots(figsize=(6, 8))

# Usa 'seismic' o 'bwr' (Blue-White-Red) para distinguir sumideros y fuentes
im = ax.imshow(csd_data, aspect='auto', cmap='seismic', 
               origin='lower', vmin=vmin, vmax=vmax)

# Añadir barra de color (colorbar)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('CSD Amplitude', fontsize=11)

# 5. ETIQUETAS Y ESTÉTICA
ax.set_xlabel('Time points / Samples', fontsize=12)
ax.set_ylabel('Channels / Depth', fontsize=12)
ax.set_title('CSD Map', fontsize=14)

plt.tight_layout()
plt.show()