# 🖱️ Virtual Mouse Using Hand Gesture Recognition

Control your computer mouse entirely with hand gestures — no physical mouse needed!  
Built with **Python**, **MediaPipe**, **OpenCV**, and **PyAutoGUI**.

---

## 📸 Demo

> Run the script and wave your hand in front of your webcam to control the cursor.

---

## 🚀 Features

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Move cursor |
| ✌️ Two fingers up | Click mode (ready state) |
| ✌️ Two fingers pinch together | Left click |
| 👌 Thumb + Index pinch | Right click |
| 🖐️ All five fingers up | Scroll (move hand up/down) |
| ✊ Fist | Click & drag |

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **OpenCV** – webcam capture & display
- **MediaPipe** – real-time hand landmark detection (21 keypoints)
- **PyAutoGUI** – OS-level mouse/keyboard control
- **NumPy** – coordinate math & smoothing

---

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/virtual-mouse.git
cd virtual-mouse

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Linux extra step
```bash
sudo apt-get install python3-tk python3-dev
```

---

## ▶️ Usage

```bash
python virtual_mouse.py
```

- A window opens showing the webcam feed with hand skeleton overlay.
- The **gesture label** is displayed at the top-left corner.
- Press **`q`** to quit.

---

## 📁 Project Structure

```
virtual-mouse/
├── virtual_mouse.py    # Main application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🔧 Configuration

Edit the constants at the top of `virtual_mouse.py`:

| Variable | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `SMOOTHING` | `5` | Cursor smoothing buffer size |
| `CLICK_THRESHOLD` | `35` | Pixel distance to trigger click |
| `SCROLL_SENSITIVITY` | `20` | Scroll speed |

---

## 🔭 How It Works

1. Webcam captures 30fps video.
2. MediaPipe detects 21 hand landmarks per frame.
3. Finger states (up/down) are determined from landmark Y-coordinates.
4. A gesture is classified based on the finger combination.
5. PyAutoGUI translates the gesture into the corresponding OS mouse action.
6. A smoothing buffer reduces jitter on cursor movement.

---

## 🌐 SDG Alignment

This project maps to **UN SDG 10 – Reduced Inequalities**, enabling people with limited motor ability to use computers via gesture-based input.

---

## 📜 License

MIT License — free to use, modify, and distribute.
