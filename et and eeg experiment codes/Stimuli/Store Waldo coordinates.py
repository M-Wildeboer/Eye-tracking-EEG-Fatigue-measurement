import os
import csv
import time
import tkinter as tk
from PIL import Image, ImageTk
from glob import glob


# ===== SETTINGS =====
STIMULI_CSV = "Stimuli.csv"
OUTPUT_CSV = "waldo_click_results.csv"

# Only show these image files
VALID_FILES = glob("*.png") + glob("*.jpg")

print(VALID_FILES)


class WaldoTask:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Search Task - Find Waldo")
        self.root.configure(bg="black")

        # Open in fullscreen mode and allow Escape to exit fullscreen
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", self.exit_fullscreen)

        self.base_folder = os.path.dirname(os.path.abspath(__file__))

        # Load the image list and prepare result storage
        self.stimuli = self.load_stimuli()
        self.results = []

        self.current_index = -1
        self.start_time = None

        self.current_file = None
        self.original_width = None
        self.original_height = None

        self.displayed_width = None
        self.displayed_height = None
        self.image_left = None
        self.image_top = None

        self.tk_img = None

        # Start screen shown before the task begins
        self.info_label = tk.Label(
            root,
            text="Klik om te starten.\nZoek Waldo en klik op de plek waar je hem vindt.",
            font=("Arial", 24),
            fg="white",
            bg="black",
            justify="center"
        )
        self.info_label.pack(expand=True)

        # Canvas used to display each stimulus image
        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.bind("<Button-1>", self.on_click)

        self.started = False

    def exit_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", False)

    def load_stimuli(self):
        stimuli = []

        csv_path = os.path.join(self.base_folder, STIMULI_CSV)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{STIMULI_CSV} niet gevonden in {self.base_folder}")

        # Read the stimulus file list from Stimuli.csv and keep only valid image files
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row["File"].strip()
                if file_name in VALID_FILES:
                    stimuli.append(file_name)

        # sort by 1.jpg, 2.jpg, ...
        #stimuli.sort(key=lambda x: int(x.split(".")[0]))

        if not stimuli:
            raise ValueError("Geen geldige stimuli gevonden in Stimuli.csv")

        return stimuli

    def start_task(self):
        self.started = True
        self.info_label.pack_forget()
        self.canvas.pack(fill="both", expand=True)
        self.show_next_image()

    def show_next_image(self):
        self.current_index += 1

        # Stop when all stimuli have been shown
        if self.current_index >= len(self.stimuli):
            self.finish_task()
            return

        self.current_file = self.stimuli[self.current_index]
        img_path = os.path.join(self.base_folder, self.current_file)

        if not os.path.exists(img_path):
            print(f"Bestand niet gevonden: {img_path}")
            self.show_next_image()
            return

        img = Image.open(img_path)
        self.original_width, self.original_height = img.size

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Scale the image to fit the screen while keeping the aspect ratio
        scale = min(screen_w / self.original_width, screen_h / self.original_height)
        self.displayed_width = int(self.original_width * scale)
        self.displayed_height = int(self.original_height * scale)

        resized = img.resize((self.displayed_width, self.displayed_height), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.config(width=screen_w, height=screen_h)

        # Center the image on the screen
        self.image_left = (screen_w - self.displayed_width) // 2
        self.image_top = (screen_h - self.displayed_height) // 2

        self.canvas.create_image(
            self.image_left,
            self.image_top,
            anchor="nw",
            image=self.tk_img
        )

        # Optional: show the file name at the top of the screen
        self.canvas.create_text(
            screen_w // 2,
            30,
            text=f"Zoek Waldo in {self.current_file}",
            fill="white",
            font=("Arial", 20)
        )

        # Start reaction time timing for this image
        self.start_time = time.perf_counter()

    def on_click(self, event):
        if not self.started:
            self.start_task()
            return

        # Check whether the click falls inside the displayed image
        x = event.x
        y = event.y

        if not (
            self.image_left <= x <= self.image_left + self.displayed_width
            and self.image_top <= y <= self.image_top + self.displayed_height
        ):
            return  # ignore clicks outside the image

        reaction_time = time.perf_counter() - self.start_time

        # Click position relative to the displayed image
        rel_x = x - self.image_left
        rel_y = y - self.image_top

        # Convert click position to x/y ratios relative to image size
        x_ratio = rel_x / self.displayed_width
        y_ratio = rel_y / self.displayed_height

        # Store the click result for this stimulus
        self.results.append({
            "file": self.current_file,
            "x_ratio": round(x_ratio, 6),
            "y_ratio": round(y_ratio, 6),
            "reaction_time_sec": round(reaction_time, 3)
        })

        self.show_next_image()

    def finish_task(self):
        output_path = os.path.join(self.base_folder, OUTPUT_CSV)

        # Save all click results to a csv file
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file", "x_ratio", "y_ratio", "reaction_time_sec"]
            )
            writer.writeheader()
            writer.writerows(self.results)

        # Show the final screen after the task is finished
        self.canvas.pack_forget()
        self.info_label.config(
            text=f"Klaar.\nResultaten opgeslagen als:\n{OUTPUT_CSV}\n\nDruk op Esc om fullscreen te verlaten en sluit daarna het venster."
        )
        self.info_label.pack(expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = WaldoTask(root)

    # The first mouse click starts the task
    root.bind("<Button-1>", lambda event: app.start_task() if not app.started else None)

    root.mainloop()