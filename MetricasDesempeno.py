import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
# Cargar X e Y
X = np.load('XCaracteristicas_Background_HOG_BaseActualizada.npy')
Y = np.load('YCaracteristicas_Background_HOG_BaseActualizada.npy')

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

# Intentar cargar el modelo, si no existe entrenarlo
model_filename = "modelo_svm_hog.pkl"

try:
    clf = joblib.load(model_filename)
    print("Modelo cargado correctamente")
except FileNotFoundError:
    print("Modelo no encontrado, entrenando uno nuevo...")

# # Predicción y evaluación
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
matriz_confusion = confusion_matrix(y_test, y_pred)
print("Matriz de confusión")
print(matriz_confusion)
precision = precision_score(y_test, y_pred)
print("Precisión: ", precision)
recall = recall_score(y_test, y_pred)
print("Sensibilidad: ", recall)
f1score = f1_score(y_test, y_pred)
print("F1 score: ", f1score)

reporte = classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"])
print("Reporte de Clasificación:")
print(reporte)



disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusion, display_labels=clf.classes_)
# Personalizar el gráfico
fig, ax = plt.subplots(figsize=(6, 6))  # Ajusta el tamaño del gráfico
disp.plot(ax=ax, cmap='Blues', colorbar=True)  # Usa un colormap para hacerlo más visual

# Ajustes adicionales (opcional)
plt.title("Matriz de Confusión")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.tight_layout()  # Evita que los elementos se superpongan

# Mostrar el gráfico
plt.show()