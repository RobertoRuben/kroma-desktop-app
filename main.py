import argparse
import os
from pathlib import Path

import cv2
import torch
from dotenv import load_dotenv
from ultralytics import solutions

load_dotenv()


def load_env_config():
    """Carga configuraciones para procesamiento de video local."""
    return {
        "VIDEO_FILENAME": os.getenv("VIDEO_FILENAME", "20260205_173719.mp4"),
        "MODEL_CONFIDENCE": float(os.getenv("MODEL_CONFIDENCE", "0.40")),
        "WINDOW_WIDTH": 1280,
        "WINDOW_HEIGHT": 720,
    }


def apply_rotation(frame, rotation_code):
    """Aplica rotación al frame si es necesario."""
    if rotation_code == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_code == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_code == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Pepper Counter - Conteo de objetos con YOLO + Tracking"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Ruta al archivo de video (opcional, sobreescribe .env)",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="vertical",
        choices=["horizontal", "vertical"],
        help="vertical: objetos se mueven de izq a der. horizontal: objetos de arriba a abajo",
    )
    parser.add_argument(
        "--roi-side",
        type=str,
        default=None,
        choices=["left", "right", "top", "bottom"],
        help="Lado donde aparece la línea de conteo. "
        "Para orientación vertical: left o right (default: right). "
        "Para orientación horizontal: top o bottom (default: bottom).",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotar video si fue grabado en vertical",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Guardar video procesado con anotaciones (default: True)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="No mostrar ventana de visualización (solo procesar y guardar)",
    )

    args = parser.parse_args()
    config = load_env_config()

    # 1. Configuración de Dispositivo (GPU Check)
    device = "0" if torch.cuda.is_available() else "cpu"
    device_name = "GPU (CUDA)" if device == "0" else "CPU"
    print(f"🚀 Dispositivo: {device_name}")

    # 2. Rutas y Carga de Video
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best.pt")
    tracker_config = os.path.join(base_dir, "botsort.yaml")

    # Verificar modelo
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra el modelo: {model_path}")
        print("   Coloca tu modelo entrenado como 'best.pt' en la raíz del proyecto.")
        return

    # Verificar tracker config (usar default de ultralytics si no existe)
    if not os.path.exists(tracker_config):
        tracker_config = "botsort.yaml"  # Usa el default de ultralytics

    # Determinar ruta del video
    video_path = (
        args.video if args.video else os.path.join(base_dir, config["VIDEO_FILENAME"])
    )

    if not os.path.exists(video_path):
        print(f"❌ Error: No se encuentra el archivo: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error al abrir el video.")
        return

    # Obtener propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Leer primer frame para dimensiones (considerando rotación)
    ret, frame = cap.read()
    if not ret:
        print("❌ El video parece estar vacío o dañado.")
        return

    frame = apply_rotation(frame, args.rotation)
    h, w = frame.shape[:2]

    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"   Resolución: {w}x{h} | FPS: {fps} | Frames: {total_frames}")
    print(f"   Orientación: {args.orientation} | Rotación: {args.rotation}°")

    # 3. Definición de la LÍNEA DE INTERÉS (cruza todo el ancho/alto)
    # Mapeo de --roi-side a posición relativa (0.0 = izq/arriba, 1.0 = der/abajo)
    roi_positions = {"left": 0.15, "right": 0.85, "top": 0.15, "bottom": 0.85}

    if args.orientation == "horizontal":
        roi_side = args.roi_side or "bottom"
        if roi_side not in ("top", "bottom"):
            print(f"⚠️  --roi-side '{roi_side}' no aplica a orientación horizontal, usando 'bottom'")
            roi_side = "bottom"
        position = roi_positions[roi_side]
        line_y = int(h * position)
        region_points = [(0, line_y), (w, line_y)]
        print(f"📏 Línea horizontal en Y={line_y} ({roi_side}, pos={position:.0%})")
    else:
        roi_side = args.roi_side or "right"
        if roi_side not in ("left", "right"):
            print(f"⚠️  --roi-side '{roi_side}' no aplica a orientación vertical, usando 'right'")
            roi_side = "right"
        position = roi_positions[roi_side]
        line_x = int(w * position)
        region_points = [(line_x, 0), (line_x, h)]
        print(f"📏 Línea vertical en X={line_x} ({roi_side}, pos={position:.0%})")

    # 4. Inicializar ObjectCounter con Tracking (BoT-SORT)
    counter = solutions.ObjectCounter(
        model=model_path,
        region=region_points,
        show=False,  # Controlamos la ventana manualmente
        classes=[0],  # Clase a detectar (ajustar según el modelo)
        conf=config["MODEL_CONFIDENCE"],
        device=device,
        tracker=tracker_config,  # Tracker BoT-SORT para tracking robusto
        line_width=2,
        show_in=True,
        show_out=True,
    )

    # 5. Configurar escritura de video de salida
    video_writer = None
    if args.save:
        output_name = Path(video_path).stem + "_counted.mp4"
        output_path = os.path.join(base_dir, output_name)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"💾 Video de salida: {output_name}")

    # Ventana de visualización
    if not args.no_display:
        window_name = "Pepper Counter - Conteo por Línea"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config["WINDOW_WIDTH"], config["WINDOW_HEIGHT"])

    print(f"▶️  Procesando... (presiona 'q' para salir)")
    print("-" * 50)

    # Reiniciamos el video al principio (ya leímos un frame para dimensiones)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # Rotar si es necesario
        if args.rotation != 0:
            frame = apply_rotation(frame, args.rotation)

        # --- PROCESAMIENTO: Tracking + Conteo por línea ---
        # ObjectCounter ejecuta track() internamente, detecta cruces de línea
        # y anota el frame con bounding boxes, IDs de tracking y conteos
        results = counter(frame)

        # Obtener frame anotado desde los resultados
        # plot_im ya incluye: línea ROI, bounding boxes, IDs de tracking,
        # y conteos IN/OUT por clase (dibujados por Ultralytics)
        annotated_frame = results.plot_im

        # Progreso cada 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(
                f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) | IN: {results.in_count} | OUT: {results.out_count}"
            )

        # Guardar frame al video de salida
        if video_writer is not None:
            video_writer.write(annotated_frame)

        # Mostrar en ventana
        if not args.no_display:
            cv2.imshow(window_name, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⏹️  Detenido por el usuario.")
                break

    # Limpieza
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    # Resumen final
    print("-" * 50)
    print(f"✅ Proceso terminado.")
    print(f"   📊 Resultados finales:")
    print(f"      IN (cruzaron →):  {counter.in_count}")
    print(f"      OUT (cruzaron ←): {counter.out_count}")
    print(f"      TOTAL:            {counter.in_count + counter.out_count}")
    print(f"   📹 Frames procesados: {frame_count}/{total_frames}")
    if args.save:
        print(f"   💾 Video guardado en: {output_path}")

    # Conteo por clase
    if counter.classwise_count:
        print(f"\n   📋 Conteo por clase:")
        for cls_name, counts in counter.classwise_count.items():
            print(f"      {cls_name}: IN={counts['IN']} | OUT={counts['OUT']}")


if __name__ == "__main__":
    main()
