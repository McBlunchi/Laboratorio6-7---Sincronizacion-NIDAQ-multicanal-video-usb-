import threading
import queue
import modules_06_11 as modules
import config

# === Crear queues y evento de stop ===
stop_event = threading.Event()
data_queue = queue.Queue()
save_queue = queue.Queue()
playback_queue = queue.Queue()
video_queue = queue.Queue()
video_trigger_queue = queue.Queue()
trigger_queue = queue.Queue()     

# === Crear threads ===   
acq_thread = threading.Thread(
    target=modules.acquisition_thread,
    args=(data_queue, save_queue, video_trigger_queue, trigger_queue, stop_event)
)

save_thread_obj = threading.Thread(
    target=modules.save_thread,
    args=(save_queue, stop_event)
)

video_cap_thread = threading.Thread(
    target=modules.video_capture_thread,
    args=(video_queue, video_trigger_queue, stop_event)
)

video_save_thread_obj = threading.Thread(
    target=modules.video_save_thread,
    args=(video_queue, stop_event)
)

playback_thread_obj = threading.Thread(
    target=modules.playback_thread,
    args=(trigger_queue, playback_queue, stop_event)
)

# === Iniciar threads ===
acq_thread.start()
save_thread_obj.start()
playback_thread_obj.start()

if config.video_enabled:
    video_cap_thread.start()
    video_save_thread_obj.start()

# === Ejecutar plotting en el main thread ===
modules.plotting_thread(data_queue, stop_event)

# === Esperar a que terminen todos los threads ===

if config.video_enabled:
    video_cap_thread.join()
    video_save_thread_obj.join()
    
acq_thread.join()
save_thread_obj.join()
playback_thread_obj.join()

print("✅ System finished.")
