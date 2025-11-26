import os
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from player_shuttle_tracker.main_tracker import DualTracker

# ===== KONFIG: folder z modelami =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.join(SCRIPT_DIR, "models")

# jeśli pliki nazywają się dokładnie tak:
DEFAULT_PATHS = {
    "shuttle": os.path.join(MODELS_ROOT, "shuttle_detection.pt"),
    "players": os.path.join(MODELS_ROOT, "player_detection.pt"),
    "court":   os.path.join(MODELS_ROOT, "court_detection.pt"),
}

# a jeśli nie – szukamy po nazwie
FILENAME_HINTS = {
    "shuttle": re.compile(r"(shuttle|bird|lotk|badminton).*\.pt$", re.I),
    "players": re.compile(r"(player|person|people|pose).*\.pt$", re.I),
    "court":   re.compile(r"(court|line|field|pitch).*\.pt$", re.I),
}


def find_model_path(kind: str):
    """Zwraca ścieżkę do modelu typu 'shuttle', 'players', 'court' albo None."""
    default = DEFAULT_PATHS.get(kind)
    if default and os.path.isfile(default):
        return default

    if not os.path.isdir(MODELS_ROOT):
        return None

    pattern = FILENAME_HINTS[kind]
    candidates = []
    for root, _, files in os.walk(MODELS_ROOT):
        for f in files:
            if f.lower().endswith(".pt") and pattern.search(f):
                candidates.append(os.path.join(root, f))

    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    return None


class StdoutRedirect:
    """
    Przekierowanie sys.stdout:
    - wysyła logi do GUI (Text)
    - jednocześnie przepisuje je do oryginalnej konsoli
    """
    def __init__(self, app: "TrackerApp"):
        self.app = app

    def write(self, text: str):
        self.app.queue_log_from_stdout(text)

    def flush(self):
        if self.app.original_stdout:
            self.app.original_stdout.flush()


class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Badminton Tracker")
        self.root.geometry("1000x800")  # DUŻE OKNO

        # do odtwarzania wyniku
        self.last_output_path = None

        # oryginalny stdout (konsola)
        self.original_stdout = sys.stdout

        # automatyczne wykrycie modeli
        self.models = {
            "shuttle": find_model_path("shuttle"),
            "players": find_model_path("players"),
            "court":   find_model_path("court"),
        }

        # główna ramka
        main = tk.Frame(root)
        main.pack(fill="both", expand=True, padx=20, pady=10)

        # Nagłówek
        title = tk.Label(
            main,
            text="Badminton Tracking System",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=10)

        info = tk.Label(
            main,
            text="Wybierz wideo, które chcesz przeanalizować",
            font=("Arial", 10)
        )
        info.pack(pady=(0, 15))

        # --- SEKCJA WIDEO ---
        frame_video = tk.Frame(main)
        frame_video.pack(pady=5, fill="x")

        tk.Label(frame_video, text="Plik wideo wejściowy:").grid(row=0, column=0, sticky="w")
        self.video_path = tk.Entry(frame_video, width=80)
        self.video_path.grid(row=1, column=0, sticky="we", pady=2)
        tk.Button(frame_video, text="Przeglądaj…", command=self.load_video).grid(row=1, column=1, padx=5)

        frame_video.columnconfigure(0, weight=1)

        # --- SEKCJA WYJŚCIA ---
        frame_out = tk.Frame(main)
        frame_out.pack(pady=10, fill="x")

        tk.Label(frame_out, text="Zapisz wynik do:").grid(row=0, column=0, sticky="w")
        self.output_path = tk.Entry(frame_out, width=80)
        self.output_path.grid(row=1, column=0, sticky="we", pady=2)
        tk.Button(frame_out, text="Wybierz plik wyjściowy…", command=self.save_output).grid(row=1, column=1, padx=5)

        frame_out.columnconfigure(0, weight=1)

        # --- LOGI ---
        frame_logs = tk.Frame(main)
        frame_logs.pack(pady=10, fill="both", expand=True)

        tk.Label(frame_logs, text="Postęp (logi z konsoli):", anchor="w").pack(anchor="w")
        self.log_text = tk.Text(frame_logs, height=14, width=100, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        # --- PASEK PROGRESU (DETERMINATE) ---
        self.progress = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(5, 5))
        self.progress["value"] = 0

        # --- PRZYCISKI NA DOLE ---
        btn_frame = tk.Frame(main)
        btn_frame.pack(pady=5)

        self.btn_run = tk.Button(
            btn_frame, text="URUCHOM ANALIZĘ",
            font=("Arial", 12, "bold"),
            command=self.run_tracking_thread
        )
        self.btn_run.grid(row=0, column=0, padx=10)

        self.btn_play = tk.Button(
            btn_frame, text="Odtwórz wynik",
            font=("Arial", 11),
            command=self.play_output,
            state="disabled"
        )
        self.btn_play.grid(row=0, column=1, padx=10)

    # ======= GUI helpers =======
    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Pliki wideo", "*.mp4;*.avi;*.mov;*.mkv"), ("Wszystkie pliki", "*.*")]
        )
        if path:
            self.video_path.delete(0, tk.END)
            self.video_path.insert(0, path)

    def save_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("Pliki MP4", "*.mp4"), ("Wszystkie pliki", "*.*")]
        )
        if path:
            self.output_path.delete(0, tk.END)
            self.output_path.insert(0, path)

    def append_textbox(self, widget: tk.Text, msg: str):
        # blokujemy edycję przez użytkownika, ale pozwalamy programowi dopisywać
        widget.configure(state="normal")
        widget.insert(tk.END, msg + "\n")
        widget.see(tk.END)
        widget.configure(state="disabled")

    # ======= LOGI I PROGRES Z sys.stdout =======
    def queue_log_from_stdout(self, text: str):
        """
        Wywoływane z wątku roboczego przez StdoutRedirect.write().
        Tutaj:
        - przekazujemy tekst do prawdziwej konsoli
        - wrzucamy go do GUI (główny wątek przez .after)
        """
        if self.original_stdout:
            self.original_stdout.write(text)

        def _update():
            for line in text.splitlines():
                line = line.rstrip()
                if not line:
                    continue
                self.append_textbox(self.log_text, line)
                self.update_progress_from_log(line)

        self.root.after(0, _update)

    def update_progress_from_log(self, line: str):
        """
        Ustawienie paska progresu na podstawie komunikatów z main_tracker.py
        """
        mapping = [
            ("=== WCZYTYWANIE VIDEO", 5),
            ("KROK 0", 15),
            ("KROK 1", 35),
            ("KROK 2", 55),
            ("KROK 3", 75),
            ("KROK 4", 90),
            ("Wideo zapisano", 100),
            ("gotowe", 100),
        ]
        for key, value in mapping:
            if key in line:
                self.set_progress(value)
                break

    def set_progress(self, value: int):
        value = max(0, min(100, int(value)))
        self.progress["value"] = value
        self.root.update_idletasks()

    def log(self, msg: str):
        # własne logi GUI (nie z sys.stdout)
        def _():
            self.append_textbox(self.log_text, msg)
        self.root.after(0, _)

    # ======= ANALIZA =======
    def run_tracking_thread(self):
        # uruchamiamy wątkiem, plus prog i blokada przycisku
        self.btn_run.config(state="disabled")
        self.btn_play.config(state="disabled")
        self.set_progress(0)

        # przekierowanie stdout -> GUI
        sys.stdout = StdoutRedirect(self)

        threading.Thread(target=self.run_tracking, daemon=True).start()

    def run_tracking(self):
        video = self.video_path.get().strip()
        output = self.output_path.get().strip()

        missing = []
        if not video or not os.path.isfile(video):
            missing.append("plik wideo")
        if not output:
            missing.append("ścieżka wyjściowa")
        if not self.models["shuttle"]:
            missing.append("model lotki (.pt)")
        if not self.models["players"]:
            missing.append("model graczy (.pt)")

        if missing:
            sys.stdout = self.original_stdout
            self.btn_run.config(state="normal")
            messagebox.showerror(
                "Brakujące dane",
                "Uzupełnij: " + ", ".join(missing) +
                "\n\nSprawdź, czy w folderze 'models' masz wagi .pt "
                "(np. shuttle_detection.pt, player_detection.pt, court_detection.pt)."
            )
            return

        self.log("Start analizy…")
        self.set_progress(1)

        # kort opcjonalny – jeśli brak, przekazujemy None
        court_path = self.models["court"]

        try:
            tracker = DualTracker(
                shuttlecock_model_path=self.models["shuttle"],
                player_model_path=self.models["players"],
                court_model_path=court_path,  # może być None
            )
            tracker.process_video(video, output)

            self.last_output_path = output
            self.set_progress(100)
            self.btn_play.config(state="normal")
            messagebox.showinfo("Gotowe", "Analiza zakończona pomyślnie.")
        except Exception as e:
            self.log(f"Błąd: {e}")
            messagebox.showerror("Błąd", str(e))
        finally:
            # przywrócenie stdout i przycisku
            sys.stdout = self.original_stdout
            self.btn_run.config(state="normal")

    # ======= ODTWARZANIE WYNIKU =======
    def play_output(self):
        if not self.last_output_path or not os.path.isfile(self.last_output_path):
            messagebox.showerror("Błąd", "Brak pliku wynikowego do odtworzenia.")
            return

        self.log(f"Odtwarzanie wyniku: {self.last_output_path}")
        cap = cv2.VideoCapture(self.last_output_path)
        if not cap.isOpened():
            messagebox.showerror("Błąd", "Nie udało się otworzyć wideo wynikowego.")
            return

        window_name = "Wynik analizy (naciśnij 'q' aby zamknąć)"
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(window_name, frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()
