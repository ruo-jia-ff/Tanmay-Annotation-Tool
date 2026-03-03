import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import threading
import glob
import os

class ClassMappingDialog(tk.Toplevel):
    def __init__(self, parent, model_names, workspace_classes):
        super().__init__(parent)
        self.title("Select Class and Mapping")
        self.geometry("450x300")
        self.transient(parent)
        self.grab_set()
        
        self.result = None
        
        # UI Elements
        tk.Label(self, text="Select Model Class to Predict:").pack(pady=(10, 2))
        self.model_class_var = tk.StringVar(self)
        self.model_class_cb = ttk.Combobox(self, textvariable=self.model_class_var, state="readonly", width=40)
        self.model_class_cb['values'] = [f"{k}: {v}" for k, v in model_names.items()]
        if self.model_class_cb['values']:
            self.model_class_cb.current(0)
        self.model_class_cb.pack(pady=5)
        
        tk.Label(self, text="Map to Workspace Class:").pack(pady=(15, 2))
        self.ws_class_var = tk.StringVar(self)
        self.ws_class_cb = ttk.Combobox(self, textvariable=self.ws_class_var, state="readonly", width=40)
        self.ws_class_cb['values'] = [f"{i}: {c}" for i, c in enumerate(workspace_classes)]
        if self.ws_class_cb['values']:
            self.ws_class_cb.current(0)
        self.ws_class_cb.pack(pady=5)
        
        tk.Label(self, text="Other model predictions will be ignored.", fg="gray").pack(pady=5)
        
        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="Start Annotation", width=15, bg="lightblue", command=self.on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Cancel", width=15, command=self.on_cancel).pack(side=tk.LEFT, padx=10)
        
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        
    def on_ok(self):
        if not self.model_class_cb.get() or not self.ws_class_cb.get():
            return
            
        model_idx = int(self.model_class_cb.get().split(":")[0])
        ws_idx = int(self.ws_class_cb.get().split(":")[0])
        
        self.result = (model_idx, ws_idx)
        self.destroy()
        
    def on_cancel(self):
        self.result = None
        self.destroy()

COLORS = [
    "red", "blue", "green", "orange", "purple", "cyan", "magenta", 
    "yellow", "brown", "pink", "lime", "teal", "navy", "maroon", "olive"
]

class YOLOAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Annotation Tool")
        self.root.geometry("1000x700")
        
        # Application state
        self.images_dir = ""
        self.labels_dir = ""
        self.classes_file = ""
        
        self.image_files = []
        self.current_index = 0
        
        self.classes = []
        self.current_class_idx = 0
        
        # Bounding boxes for the current image:
        # dict of { 'class_idx': int, 'bbox': (x_min, y_min, x_max, y_max) } in original image coords
        self.annotations = [] 
        
        # Display state
        self.current_image = None
        self.tk_image = None
        self.img_width = 0
        self.img_height = 0
        self.scale_f = 1.0
        self.x_offset = 0
        self.y_offset = 0
        
        # Drawing state
        self.start_x = None
        self.start_y = None
        self.current_rect_id = None
        self.selected_rect_idx = None
        self.resize_handle_idx = None # 0=TL, 1=TR, 2=BL, 3=BR
        self.moving_rect = False
        self.move_start_x = None
        self.move_start_y = None
        self.orig_bbox = None
        self.canvas_rects = [] # stores canvas graphic IDs
        self.canvas_texts = []
        
        self.crosshair_h = None
        self.crosshair_v = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Left Panel (Controls)
        left_panel = tk.Frame(self.root, width=250, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        tk.Button(left_panel, text="Open Images Dir", command=self.load_images_dir, width=25).pack(pady=5)
        self.lbl_images_dir = tk.Label(left_panel, text="No Images Dir", fg="gray", bg="#f0f0f0")
        self.lbl_images_dir.pack(pady=2)
        
        tk.Button(left_panel, text="Open Labels Dir", command=self.load_labels_dir, width=25).pack(pady=5)
        self.lbl_labels_dir = tk.Label(left_panel, text="No Labels Dir", fg="gray", bg="#f0f0f0")
        self.lbl_labels_dir.pack(pady=2)
        
        tk.Button(left_panel, text="Load classes.txt", command=self.load_classes_file, width=25).pack(pady=5)
        tk.Button(left_panel, text="Auto Annotate...", command=self.auto_annotate, width=25, bg="lightblue").pack(pady=5)
        
        # Classes Listbox
        tk.Label(left_panel, text="Classes:", bg="#f0f0f0").pack(pady=(10, 0))
        self.listbox_classes = tk.Listbox(left_panel, height=8, exportselection=False)
        self.listbox_classes.pack(fill=tk.X, pady=2)
        self.listbox_classes.bind('<<ListboxSelect>>', self.on_class_select)
        
        tk.Button(left_panel, text="Add Class", command=self.add_class, width=25).pack(pady=2)
        
        # Current Annotations Listbox
        tk.Label(left_panel, text="Annotations (Select to delete/edit):", bg="#f0f0f0").pack(pady=(10, 0))
        self.listbox_annotations = tk.Listbox(left_panel, height=8, exportselection=False)
        self.listbox_annotations.pack(fill=tk.X, pady=2)
        self.listbox_annotations.bind('<<ListboxSelect>>', self.on_annotation_select)
        
        tk.Button(left_panel, text="Delete Selected (Del)", command=self.delete_selected, width=25).pack(pady=2)
        
        tk.Label(left_panel, text="Tips:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        tk.Label(left_panel, text="- Drag box edges to move it", bg="#f0f0f0").pack()
        tk.Label(left_panel, text="- Drag inside a box to draw", bg="#f0f0f0").pack()
        
        # Navigation
        nav_frame = tk.Frame(left_panel, bg="#f0f0f0")
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        tk.Button(nav_frame, text="< Prev", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next >", command=self.next_image, width=10).pack(side=tk.RIGHT, padx=5)
        
        self.lbl_progress = tk.Label(nav_frame, text="0 / 0", bg="#f0f0f0")
        self.lbl_progress.pack(side=tk.TOP, pady=5)
        
        tk.Button(left_panel, text="Save (Ctrl+S)", command=self.save_annotations, width=25, bg="lightgreen").pack(side=tk.BOTTOM, pady=5)
        
        # Right Panel (Canvas)
        self.canvas_frame = tk.Frame(self.root, bg="gray")
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Configure>", self.on_window_resize)
        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<Control-s>", lambda e: self.save_annotations())
        self.root.bind("<Key>", self.on_key_press)

    def on_key_press(self, event):
        if event.char.isdigit() and int(event.char) > 0:
            idx = int(event.char) - 1
            if idx < len(self.classes):
                self.listbox_classes.selection_clear(0, tk.END)
                self.listbox_classes.selection_set(idx)
                self.listbox_classes.event_generate("<<ListboxSelect>>")
        elif event.char.lower() == 'a':
            self.prev_image()
        elif event.char.lower() == 'd':
            self.next_image()

    def on_mouse_move(self, event):
        if not self.current_image: return
        self.canvas.delete("crosshair")
        
        self.crosshair_h = self.canvas.create_line(0, event.y, self.canvas.winfo_width(), event.y, fill="yellow", dash=(2, 2), tags="crosshair")
        self.crosshair_v = self.canvas.create_line(event.x, 0, event.x, self.canvas.winfo_height(), fill="yellow", dash=(2, 2), tags="crosshair")

    def update_class_listbox(self):
        self.listbox_classes.delete(0, tk.END)
        for i, c in enumerate(self.classes):
            self.listbox_classes.insert(tk.END, f"{i}: {c}")
        if self.classes:
            self.listbox_classes.selection_set(self.current_class_idx)

    def load_images_dir(self):
        d = filedialog.askdirectory(title="Select Images Directory")
        if d:
            self.images_dir = d
            self.lbl_images_dir.config(text=os.path.basename(self.images_dir))
            valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
            self.image_files = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in valid_exts]
            self.image_files.sort()
            self.current_index = 0
            
            if not self.labels_dir:
                self.labels_dir = d
                self.lbl_labels_dir.config(text=os.path.basename(self.labels_dir))
            
            self.load_current_image()

    def load_labels_dir(self):
        d = filedialog.askdirectory(title="Select Labels Directory")
        if d:
            self.labels_dir = d
            self.lbl_labels_dir.config(text=os.path.basename(self.labels_dir))
            if self.image_files:
                self.load_current_image()

    def load_classes_file(self):
        f = filedialog.askopenfilename(title="Select classes.txt", filetypes=[("Text Files", "*.txt")])
        if f:
            self.classes_file = f
            with open(f, 'r') as file:
                self.classes = [line.strip() for line in file if line.strip()]
            self.current_class_idx = 0
            self.update_class_listbox()

    def add_class(self):
        new_class = simpledialog.askstring("Add Class", "Enter new class name:")
        if new_class and new_class.strip():
            self.classes.append(new_class.strip())
            self.update_class_listbox()
            self.save_classes()

    def save_classes(self):
        # Auto-save classes.txt to Labels dir if not explicitly loaded
        path = self.classes_file if self.classes_file else os.path.join(self.labels_dir, "classes.txt")
        if path and os.path.isdir(os.path.dirname(path)) if path else False:
            self.classes_file = path
            with open(path, 'w') as f:
                f.write("\n".join(self.classes))

    def on_class_select(self, event):
        sel = self.listbox_classes.curselection()
        if sel:
            self.current_class_idx = sel[0]
            
            # If an annotation is selected, update its class
            if self.selected_rect_idx is not None:
                self.annotations[self.selected_rect_idx]['class_idx'] = self.current_class_idx
                self.redraw_annotations()
                self.save_annotations()

    def on_annotation_select(self, event):
        sel = self.listbox_annotations.curselection()
        if sel:
            self.selected_rect_idx = sel[0]
        else:
            self.selected_rect_idx = None
        self.redraw_annotations()

    def delete_selected(self):
        if self.selected_rect_idx is not None:
            del self.annotations[self.selected_rect_idx]
            self.selected_rect_idx = None
            self.redraw_annotations()
            self.save_annotations()

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.save_annotations()
            self.current_index += 1
            self.load_current_image()

    def prev_image(self):
        if self.image_files and self.current_index > 0:
            self.save_annotations()
            self.current_index -= 1
            self.load_current_image()

    def get_current_label_path(self):
        if not self.image_files or not self.labels_dir:
            return None
        img_name = self.image_files[self.current_index]
        base_name = os.path.splitext(img_name)[0]
        return os.path.join(self.labels_dir, base_name + ".txt")

    def load_current_image(self):
        if not self.image_files:
            return
            
        self.annotations = []
        self.selected_rect_idx = None
        
        img_path = os.path.join(self.images_dir, self.image_files[self.current_index])
        try:
            self.current_image = Image.open(img_path)
            self.img_width, self.img_height = self.current_image.size
            self.lbl_progress.config(text=f"{self.current_index + 1} / {len(self.image_files)}")
            
            # Load annotations
            label_path = self.get_current_label_path()
            if label_path and os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            c_idx = int(parts[0])
                            # YOLO format: x_center, y_center, width, height (normalized)
                            xc, yc, w, h = map(float, parts[1:5])
                            
                            x_min = (xc - w/2) * self.img_width
                            y_min = (yc - h/2) * self.img_height
                            x_max = (xc + w/2) * self.img_width
                            y_max = (yc + h/2) * self.img_height
                            
                            self.annotations.append({'class_idx': c_idx, 'bbox': (x_min, y_min, x_max, y_max)})
                            
            self.display_image()
        except Exception as e:
            messagebox.showerror("Error load_current_image", str(e))

    def display_image(self):
        if not self.current_image:
            return
            
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        
        if c_width <= 1 or c_height <= 1:
            # Not drawn yet, try updating
            self.root.update_idletasks()
            c_width = max(2, self.canvas.winfo_width())
            c_height = max(2, self.canvas.winfo_height())
            
        img_w, img_h = self.current_image.size
        
        self.scale_f = min(c_width / img_w, c_height / img_h)
        new_w = int(img_w * self.scale_f)
        new_h = int(img_h * self.scale_f)
        
        self.x_offset = (c_width - new_w) // 2
        self.y_offset = (c_height - new_h) // 2
        
        resized_img = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.x_offset, self.y_offset, anchor=tk.NW, image=self.tk_image)
        
        self.redraw_annotations()

    def on_window_resize(self, event):
        if event.widget == self.root and self.current_image:
            # Using after to debounce
            if hasattr(self, '_resize_job'):
                self.root.after_cancel(self._resize_job)
            self._resize_job = self.root.after(200, self.display_image)

    def redraw_annotations(self):
        self.canvas.delete("bbox")
        self.canvas.delete("handle")
        self.listbox_annotations.delete(0, tk.END)
        
        for i, ann in enumerate(self.annotations):
            cls_idx = ann['class_idx']
            x_min, y_min, x_max, y_max = ann['bbox']
            
            # transform to canvas coordinates
            cx1 = x_min * self.scale_f + self.x_offset
            cy1 = y_min * self.scale_f + self.y_offset
            cx2 = x_max * self.scale_f + self.x_offset
            cy2 = y_max * self.scale_f + self.y_offset
            
            cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else f"Unknown ({cls_idx})"
            self.listbox_annotations.insert(tk.END, f"{cls_name} ({int(x_min)}, {int(y_min)})")
            
            base_color = COLORS[cls_idx % len(COLORS)]
            color = base_color if i != self.selected_rect_idx else "white"
            dash = () if i != self.selected_rect_idx else (4, 4)
            width = 5 if i == self.selected_rect_idx else 3
            
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width, tags="bbox", dash=dash)
            
            # Make label text more visible with a background rectangle
            text_id = self.canvas.create_text(cx1, cy1 - 10, text=cls_name, fill="white", anchor=tk.W, font=("Arial", 12, "bold"), tags="bbox")
            text_bbox = self.canvas.bbox(text_id)
            if text_bbox:
                bg_id = self.canvas.create_rectangle(text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2, fill=base_color, outline=base_color, tags="bbox")
                self.canvas.tag_lower(bg_id, text_id)
            
            # Draw resize handles if selected
            if i == self.selected_rect_idx:
                h_size = 4
                corners = [
                    (cx1, cy1), # TL
                    (cx2, cy1), # TR
                    (cx1, cy2), # BL
                    (cx2, cy2)  # BR
                ]
                for cx, cy in corners:
                    self.canvas.create_rectangle(cx-h_size, cy-h_size, cx+h_size, cy+h_size, fill="white", outline="black", tags="handle")
            
        if self.selected_rect_idx is not None:
            self.listbox_annotations.selection_set(self.selected_rect_idx)

    def _get_image_coords(self, canvas_x, canvas_y):
        img_x = (canvas_x - self.x_offset) / self.scale_f
        img_y = (canvas_y - self.y_offset) / self.scale_f
        # clamp
        img_x = max(0, min(img_x, self.img_width))
        img_y = max(0, min(img_y, self.img_height))
        return img_x, img_y

    def on_button_press(self, event):
        if not self.current_image: return
        
        img_x, img_y = self._get_image_coords(event.x, event.y)
        
        self.resize_handle_idx = None
        self.moving_rect = False
        
        # Check if clicked on a handle of the selected rect
        if self.selected_rect_idx is not None:
            ann = self.annotations[self.selected_rect_idx]
            x_min, y_min, x_max, y_max = ann['bbox']
            
            # Handle hit detection
            h_size = 5
            corners = [
                (x_min, y_min), # TL 0
                (x_max, y_min), # TR 1
                (x_min, y_max), # BL 2
                (x_max, y_max)  # BR 3
            ]
            for idx, (cx, cy) in enumerate(corners):
                canvas_cx = cx * self.scale_f + self.x_offset
                canvas_cy = cy * self.scale_f + self.y_offset
                if abs(event.x - canvas_cx) <= h_size * 2 and abs(event.y - canvas_cy) <= h_size * 2:
                    self.resize_handle_idx = idx
                    self.start_x = img_x
                    self.start_y = img_y
                    return # start resizing
        
        # Check if Shift is held down (event.state & 0x0001) for forcing draw (optional now)
        force_draw = bool(event.state & 0x0001)
        
        # Check if clicked on outline of existing bbox
        clicked_idx = None
        if not force_draw:
            margin = 8 / self.scale_f  # 8 pixels tolerance for outline
            for i in range(len(self.annotations)-1, -1, -1):
                ann = self.annotations[i]
                x_min, y_min, x_max, y_max = ann['bbox']
                
                near_x = min(abs(img_x - x_min), abs(img_x - x_max)) <= margin
                near_y = min(abs(img_y - y_min), abs(img_y - y_max)) <= margin
                in_y_range = (y_min - margin) <= img_y <= (y_max + margin)
                in_x_range = (x_min - margin) <= img_x <= (x_max + margin)
                
                if (near_x and in_y_range) or (near_y and in_x_range):
                    clicked_idx = i
                    break
                
        if clicked_idx is not None:
            self.selected_rect_idx = clicked_idx
            self.moving_rect = True
            self.move_start_x = img_x
            self.move_start_y = img_y
            self.orig_bbox = self.annotations[clicked_idx]['bbox']
            self.redraw_annotations()
            return
            
        if not self.classes:
            messagebox.showwarning("Warning", "Please add or load classes first!")
            return
            
        # Start new drawing
        self.selected_rect_idx = None
        self.start_x = img_x
        self.start_y = img_y
        self.current_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=COLORS[self.current_class_idx % len(COLORS)], width=2, tags="bbox")

    def on_mouse_drag(self, event):
        if not self.current_image: return
        
        img_x, img_y = self._get_image_coords(event.x, event.y)
        self.on_mouse_move(event) # update crosshair
        
        if self.resize_handle_idx is not None and self.selected_rect_idx is not None:
            ann = self.annotations[self.selected_rect_idx]
            x_min, y_min, x_max, y_max = ann['bbox']
            
            # modify bbox based on handle
            if self.resize_handle_idx == 0: x_min, y_min = img_x, img_y
            elif self.resize_handle_idx == 1: x_max, y_min = img_x, img_y
            elif self.resize_handle_idx == 2: x_min, y_max = img_x, img_y
            elif self.resize_handle_idx == 3: x_max, y_max = img_x, img_y
                
            # Avoid bounds flipping permanently during drag by ordering properly on release, but for rendering:
            render_xmin = min(x_min, x_max)
            render_xmax = max(x_min, x_max)
            render_ymin = min(y_min, y_max)
            render_ymax = max(y_min, y_max)
            
            self.annotations[self.selected_rect_idx]['bbox'] = (x_min, y_min, x_max, y_max)
            self.redraw_annotations()
            
        elif getattr(self, 'moving_rect', False) and self.selected_rect_idx is not None:
            # Move the entire bbox
            dx = img_x - self.move_start_x
            dy = img_y - self.move_start_y
            
            ox_min, oy_min, ox_max, oy_max = self.orig_bbox
            
            new_xmin = ox_min + dx
            new_ymin = oy_min + dy
            new_xmax = ox_max + dx
            new_ymax = oy_max + dy
            
            # Clamp to image boundaries
            if new_xmin < 0:
                new_xmax -= new_xmin
                new_xmin = 0
            if new_ymin < 0:
                new_ymax -= new_ymin
                new_ymin = 0
            if new_xmax > self.img_width:
                new_xmin -= (new_xmax - self.img_width)
                new_xmax = self.img_width
            if new_ymax > self.img_height:
                new_ymin -= (new_ymax - self.img_height)
                new_ymax = self.img_height
                
            self.annotations[self.selected_rect_idx]['bbox'] = (new_xmin, new_ymin, new_xmax, new_ymax)
            self.redraw_annotations()

        elif self.current_rect_id is not None:
            # Update canvas coords
            self.canvas.coords(self.current_rect_id, 
                               self.start_x * self.scale_f + self.x_offset, 
                               self.start_y * self.scale_f + self.y_offset, 
                               event.x, event.y)

    def on_button_release(self, event):
        if self.resize_handle_idx is not None and self.selected_rect_idx is not None:
            # Normalize bounding box coords after resize
            ann = self.annotations[self.selected_rect_idx]
            x_min, y_min, x_max, y_max = ann['bbox']
            x_min, x_max = min(x_min, x_max), max(x_min, x_max)
            y_min, y_max = min(y_min, y_max), max(y_min, y_max)
            self.annotations[self.selected_rect_idx]['bbox'] = (x_min, y_min, x_max, y_max)
            
            self.resize_handle_idx = None
            self.save_annotations()
            return
            
        if getattr(self, 'moving_rect', False):
            self.moving_rect = False
            self.save_annotations()
            return
            
        if self.current_rect_id is not None:
            end_x, end_y = self._get_image_coords(event.x, event.y)
            
            x_min = min(self.start_x, end_x)
            y_min = min(self.start_y, end_y)
            x_max = max(self.start_x, end_x)
            y_max = max(self.start_y, end_y)
            
            # Reject tiny boxes
            if (x_max - x_min) > 5 and (y_max - y_min) > 5:
                self.annotations.append({
                    'class_idx': self.current_class_idx,
                    'bbox': (x_min, y_min, x_max, y_max)
                })
                self.selected_rect_idx = len(self.annotations) - 1
                
            self.current_rect_id = None
            self.redraw_annotations()
            self.save_annotations()

    def save_annotations(self):
        label_path = self.get_current_label_path()
        if not label_path: return
        
        if not self.annotations:
            # Remove file if no annotations
            if os.path.exists(label_path):
                os.remove(label_path)
            return
            
        try:
            with open(label_path, 'w') as f:
                for ann in self.annotations:
                    x_min, y_min, x_max, y_max = ann['bbox']
                    
                    # YOLO conversion
                    w = (x_max - x_min) / self.img_width
                    h = (y_max - y_min) / self.img_height
                    xc = (x_min / self.img_width) + (w / 2)
                    yc = (y_min / self.img_height) + (h / 2)
                    
                    f.write(f"{ann['class_idx']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            messagebox.showerror("Error save_annotations", str(e))

    def auto_annotate(self):
        if not self.classes:
            messagebox.showwarning("Warning", "Please add or load classes first before auto-annotating, so we have something to map to!")
            return
            
        model_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if not model_path: return
            
        target_dir = filedialog.askdirectory(title="Select Directory to Auto-Annotate")
        if not target_dir: return
        
        try:
            from ultralytics import YOLO
        except ImportError:
            messagebox.showerror("Import Error", "ultralytics package is required. Run 'pip install ultralytics'")
            return
            
        # Load model first to get its class names
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Loading...")
        progress_win.geometry("250x80")
        progress_win.transient(self.root)
        tk.Label(progress_win, text="Loading YOLO model names...").pack(pady=20)
        self.root.update()
        
        try:
            model = YOLO(model_path)
            model_names = model.names
        except Exception as e:
            progress_win.destroy()
            messagebox.showerror("Model Error", str(e))
            return
            
        progress_win.destroy()
        
        # Ask for mapping
        dialog = ClassMappingDialog(self.root, model_names, self.classes)
        self.root.wait_window(dialog)
        
        if not dialog.result:
            return
            
        target_model_cls_idx, target_ws_cls_idx = dialog.result
        
        progress_win = tk.Toplevel(self.root)
        progress_win.title("Auto Annotating...")
        progress_win.geometry("400x120")
        progress_win.transient(self.root)
        progress_win.grab_set()
        
        lbl_info = tk.Label(progress_win, text="Loading model...")
        lbl_info.pack(pady=10)
        
        pb = ttk.Progressbar(progress_win, mode='determinate', length=300)
        pb.pack(pady=10)
        
        def run_inference():
            try:
                model = YOLO(model_path)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Model Error", str(e)))
                self.root.after(0, progress_win.destroy)
                return
                
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            img_paths = []
            for root_dir, dirs, files in os.walk(target_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        img_paths.append(os.path.join(root_dir, f))
                        
            total = len(img_paths)
            if total == 0:
                self.root.after(0, lambda: messagebox.showinfo("Done", "No images found."))
                self.root.after(0, progress_win.destroy)
                return
                
            pb.configure(maximum=total)
            count = 0
            
            for img_path in img_paths:
                try:
                    self.root.after(0, lambda p=img_path: lbl_info.config(text=f"Processing: {os.path.basename(p)}"))
                    results = model(img_path, verbose=False)
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    
                    # Read existing annotations if any to keep them, we will append new ones
                    existing_lines = []
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r') as ext_f:
                            existing_lines = ext_f.readlines()
                    
                    new_boxes = []
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            # ONLY ADD IF IT MATCHES THE SELECTED CLASS
                            if cls_id == target_model_cls_idx:
                                x, y, w, h = box.xywhn[0].tolist()
                                new_boxes.append(f"{target_ws_cls_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                                
                    if existing_lines or new_boxes:
                        with open(txt_path, 'w') as out_f:
                            for ln in existing_lines:
                                out_f.write(ln)
                            for nb in new_boxes:
                                out_f.write(nb)
                                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                
                count += 1
                self.root.after(0, lambda c=count: pb.configure(value=c))
                            
            def on_success():
                messagebox.showinfo("Success", f"Auto-annotated {count} images! Added mapped boxes for {model_names[target_model_cls_idx]} -> {self.classes[target_ws_cls_idx]}.")
                # Automatically open the target dir in the tool
                self.images_dir = target_dir
                self.labels_dir = target_dir
                self.lbl_images_dir.config(text=os.path.basename(self.images_dir))
                self.lbl_labels_dir.config(text=os.path.basename(self.labels_dir))
                
                # We shouldn't overwrite the workspace classes with the model classes anymore
                # because we are mapping model classes TO workspace classes.
                # So we remove the classes_path loading from the auto_annotate target dir
                # instead we ensure we just reload the images.
                
                # Load all images recursively with relative paths
                valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                all_rel_paths = []
                for root_dir, _, files in os.walk(target_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in valid_exts:
                            abs_path = os.path.join(root_dir, f)
                            rel_path = os.path.relpath(abs_path, target_dir)
                            all_rel_paths.append(rel_path)
                
                self.image_files = sorted(all_rel_paths)
                self.current_index = 0
                
                if self.image_files:
                    self.load_current_image()
                    
                progress_win.destroy()
                
            self.root.after(0, on_success)
            
        threading.Thread(target=run_inference, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOAnnotationTool(root)
    root.mainloop()
