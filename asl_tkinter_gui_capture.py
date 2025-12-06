"""
ASL Recognizer – Upload + Webcam Capture
Added:
 - Delete last character button
 - Clear whole sequence button
"""

import threading
import time
import os
import io
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# ---------------- TensorFlow Setup ----------------
tf.compat.v1.disable_eager_execution()

GRAPH_PATH = "logs/output_graph.pb"
LABELS_PATH = "logs/output_labels.txt"

if not os.path.exists(GRAPH_PATH) or not os.path.exists(LABELS_PATH):
    raise SystemExit("Missing logs/output_graph.pb or output_labels.txt")

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [l.strip() for l in f.readlines()]

# Load TF graph
with tf.io.gfile.GFile(GRAPH_PATH, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

_graph = tf.Graph()
with _graph.as_default():
    tf.import_graph_def(graph_def, name="")

sess = tf.compat.v1.Session(graph=_graph)
_graph_lock = threading.Lock()

# Get final output tensor
try:
    softmax_tensor = sess.graph.get_tensor_by_name("final_result:0")
except:
    for op in sess.graph.get_operations():
        if op.type == "Softmax":
            softmax_tensor = sess.graph.get_tensor_by_name(op.name + ":0")
            break

if softmax_tensor is None:
    raise SystemExit("Softmax output not found.")


# ---------------- Prediction function ----------------
def predict_from_bytes(image_bytes, top_k=5):
    with _graph_lock:
        preds = sess.run(softmax_tensor, {"DecodeJpeg/contents:0": image_bytes})
    preds = np.squeeze(preds)
    idxs = preds.argsort()[-top_k:][::-1]
    return [(labels[i], float(preds[i])) for i in idxs]


# ---------------- Tkinter App ----------------
class ASLApp:
    def __init__(self, root):
        self.root = root
        root.title("Hand Gesture Recognizer — Upload + Webcam Capture")
        root.geometry("1200x780")

        # GLOBAL sequence
        self.sequence = ""

        # ==== LEFT PANEL ====
        left = tk.Frame(root, width=350)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # -------- Upload Section --------
        tk.Label(left, text="Upload & Predict", font=("Arial", 14, "bold")).pack()

        self.btn_upload = tk.Button(left, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack(fill=tk.X, pady=5)

        self.btn_predict_img = tk.Button(left, text="Predict Uploaded Image",
                                         state=tk.DISABLED,
                                         command=self.predict_uploaded)
        self.btn_predict_img.pack(fill=tk.X)

        tk.Label(left, text="Top-5 Predictions:").pack(anchor="w", pady=(10, 0))
        self.list_preds = tk.Listbox(left, height=7)
        self.list_preds.pack(fill=tk.X)

        # -------- Webcam Section --------
        tk.Label(left, text="Webcam Controls", font=("Arial", 14, "bold")).pack(pady=10)

        self.btn_webcam = tk.Button(left, text="Open Webcam", command=self.toggle_webcam)
        self.btn_webcam.pack(fill=tk.X, pady=5)

        self.btn_capture = tk.Button(left, text="Capture ROI",
                                     state=tk.DISABLED,
                                     command=self.capture_roi)
        self.btn_capture.pack(fill=tk.X)

        # -------- Sequence Controls --------
        tk.Label(left, text="Sequence:").pack(anchor="w", pady=(10, 0))
        self.sequence_var = tk.StringVar(value="")
        self.entry_seq = tk.Entry(left, textvariable=self.sequence_var, font=("Arial", 12))
        self.entry_seq.pack(fill=tk.X)

        # NEW BUTTONS → Delete + Clear
        btn_frame = tk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=6)

        self.btn_delete = tk.Button(btn_frame, text="Delete Last", command=self.delete_last)
        self.btn_delete.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear_sequence)
        self.btn_clear.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # -------- Status --------
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(left, textvariable=self.status_var, fg="blue").pack(pady=10)

        # ==== RIGHT PANEL ====
        right = tk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Preview (upload or captured ROI)
        self.lbl_preview = tk.Label(right, bg="black")
        self.lbl_preview.pack(fill=tk.BOTH, expand=True)

        # Webcam area
        self.lbl_webcam = tk.Label(right, bg="black")
        self.lbl_webcam.pack(fill=tk.X, expand=True, pady=5)

        # ---- Internal vars ----
        self.uploaded_bytes = None
        self.uploaded_imgtk = None

        self._webcam_running = False
        self._cap = None
        self._last_frame = None

    # ==================== DELETE BUTTON ====================
    def delete_last(self):
        if self.sequence:
            self.sequence = self.sequence[:-1]
            self.sequence_var.set(self.sequence)
            self.status_var.set("Deleted last character.")

    # ==================== CLEAR BUTTON ====================
    def clear_sequence(self):
        self.sequence = ""
        self.sequence_var.set("")
        self.status_var.set("Sequence cleared.")

    # ==================== Upload Image ====================
    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not path:
            return

        pil = Image.open(path).convert("RGB")
        pil.thumbnail((700, 700))
        self.uploaded_imgtk = ImageTk.PhotoImage(pil)

        self.lbl_preview.config(image=self.uploaded_imgtk)

        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        self.uploaded_bytes = buf.getvalue()

        self.btn_predict_img.config(state=tk.NORMAL)
        self.status_var.set("Image loaded.")

    def predict_uploaded(self):
        if not self.uploaded_bytes:
            return

        self.status_var.set("Predicting...")
        preds = predict_from_bytes(self.uploaded_bytes, top_k=5)

        self.list_preds.delete(0, tk.END)
        for lab, sc in preds:
            self.list_preds.insert(tk.END, f"{lab} (score={sc:.4f})")

        # Add best prediction to global sequence
        best_label = preds[0][0]
        if best_label == "space":
            self.sequence += " "
        elif best_label == "del":
            self.sequence = self.sequence[:-1]
        elif best_label != "nothing":
            self.sequence += best_label

        self.sequence_var.set(self.sequence)
        self.status_var.set("Upload prediction added to sequence.")

    # ==================== Webcam Controls ====================
    def toggle_webcam(self):
        if self._webcam_running:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self._webcam_running = True
        self.btn_webcam.config(text="Close Webcam")
        self.btn_capture.config(state=tk.NORMAL)

        t = threading.Thread(target=self.webcam_loop, daemon=True)
        t.start()
        self.status_var.set("Webcam Started")

    def stop_webcam(self):
        self._webcam_running = False
        if self._cap:
            self._cap.release()
        self.btn_webcam.config(text="Open Webcam")
        self.btn_capture.config(state=tk.DISABLED)
        self.status_var.set("Webcam Stopped")

    # ==================== Webcam Loop ====================
    def webcam_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            self.status_var.set("Camera not found")
            return

        self._cap = cap

        prev = ""
        same = 0
        last_added = None
        frame_id = 0

        while self._webcam_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            self._last_frame = frame.copy()

            # ROI
            x1, y1, x2, y2 = 100, 100, 300, 300
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            crop = frame[y1:y2, x1:x2]

            ok, jpg = cv2.imencode(".jpg", crop)
            if not ok:
                continue
            img_bytes = jpg.tobytes()

            # Predict every 4 frames
            if frame_id % 4 == 0:
                preds = predict_from_bytes(img_bytes, top_k=1)
                label, score = preds[0]

                # ---- stable detection logic ----
                if label == prev:
                    same += 1
                else:
                    same = 1
                    prev = label

                # add after stable detection
                if same == 3 and label not in ["nothing", "ERR"]:
                    if label != last_added:
                        if label == "space":
                            self.sequence += " "
                        elif label == "del":
                            self.sequence = self.sequence[:-1]
                        else:
                            self.sequence += label

                        last_added = label
                        self.sequence_var.set(self.sequence)

                if label == "nothing":
                    last_added = None

            frame_id += 1

            # Draw text
            cv2.putText(frame, f"{label.upper()} ({score:.2f})",
                        (10, 420), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((850, 500))
            imgtk = ImageTk.PhotoImage(pil)
            self.lbl_webcam.imgtk = imgtk
            self.lbl_webcam.config(image=imgtk)

            time.sleep(0.02)

        cap.release()

    # ==================== Capture ROI ====================
    def capture_roi(self):
        if self._last_frame is None:
            return

        x1, y1, x2, y2 = 100, 100, 300, 300
        crop = self._last_frame[y1:y2, x1:x2]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((700, 700))

        imgtk = ImageTk.PhotoImage(pil)
        self.lbl_preview.config(image=imgtk)
        self.uploaded_imgtk = imgtk

        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        preds = predict_from_bytes(img_bytes, top_k=5)

        self.list_preds.delete(0, tk.END)
        for lab, sc in preds:
            self.list_preds.insert(tk.END, f"{lab}: {sc:.4f}")

        # Add to sequence
        best = preds[0][0]
        if best == "space":
            self.sequence += " "
        elif best == "del":
            self.sequence = self.sequence[:-1]
        elif best != "nothing":
            self.sequence += best

        self.sequence_var.set(self.sequence)
        self.status_var.set("Captured ROI added to sequence.")


# ---------------- Run App ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, "_webcam_running", False), root.destroy()))
    root.mainloop()

    sess.close()
