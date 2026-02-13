from ultralytics import YOLO

# Ruta a tu modelo de pimientos
model_path = r"D:\Proyectos\pepper-counter\pepper_ripeness_cls_v1.pt"

# Cargar modelo
model = YOLO(model_path)

# Obtener nombres y ordenarlos por su ID
class_names = model.names
ordered_classes = [class_names[i] for i in sorted(class_names.keys())]

print("--- LISTA DE CLASES EN ORDEN (ÍNDICE: NOMBRE) ---")
for idx, name in enumerate(ordered_classes):
    print(f"{idx}: '{name}'")

# Exportar como lista pura por si la necesitas en otro script
print("\nComo lista de Python:")
print(ordered_classes)