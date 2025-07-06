# Script para corregir el gráfico de dispersión
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df = pd.read_csv('data/processed/datos_limpios.csv')

# Crear el gráfico de dispersión corregido
plt.figure(figsize=(12, 8))
ax = sns.scatterplot(data=df, x='Calificación', y=df['Total Venta']/1e6, hue='Tienda', palette='viridis', s=100, alpha=0.6)
plt.title('Relación entre Calificaciones e Ingresos', pad=20, fontsize=14, fontweight='bold')
plt.xlabel('Calificación', fontsize=12)
plt.ylabel('Venta Total (Millones)', fontsize=12)

# Personalizar la leyenda y los bordes
plt.legend(title='Tienda', title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Guardar el gráfico corregido
plt.savefig('visualizations/relacion_calificaciones_ingresos_corregido.png', bbox_inches='tight', dpi=300)
plt.show()

print("Gráfico corregido generado correctamente.")