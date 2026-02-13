from pymediainfo import MediaInfo

file_path = r"D:\Proyectos\pepper-counter\20260212_092659.mp4"

def get_video_details(path):
    media_info = MediaInfo.parse(path)
    video_track = None

    for track in media_info.tracks:
        if track.track_type == "Video":
            video_track = track
            break

    if video_track:
        print(f"--- Detalles de Video: {path} ---")
        print(f"Codec ID: {video_track.codec_id}")
        print(f"Formato: {video_track.format}")
        print(f"Perfil del formato: {video_track.format_profile}")
        print(f"Ancho x Alto: {video_track.width}x{video_track.height}")
        print(f"Frame Rate: {video_track.frame_rate} fps")
        
        print("\n--- Detalles de Color (Aquí suele estar el problema) ---")
        print(f"Espacio de color: {video_track.color_space}")
        print(f"Chroma subsampling: {video_track.chroma_subsampling}")
        print(f"Bit depth: {video_track.bit_depth} bits")
        print(f"Primarias de color: {video_track.color_primaries}")
        print(f"Características de transferencia: {video_track.transfer_characteristics}")
        print(f"Coeficientes de matriz: {video_track.matrix_coefficients}")
        print(f"Rango de color: {video_track.color_range}") # Importante: Limited vs Full
    else:
        print("No se encontró una pista de video en el archivo.")

get_video_details(file_path)