import cv2
import numpy as np
import time

# ---------- MLP inference-only ----------
def relu(x): return np.maximum(0.0, x)
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class MLPInference:
    def __init__(self, weight_path="weights_mlp.npz", layer_sizes=(784,128,64,10)):
        self.W = []; self.b = []
        data = np.load(weight_path, allow_pickle=False)
        arrays = [data[k] for k in data.files]
        L = len(layer_sizes) - 1
        for i in range(L):
            self.W.append(arrays[2*i])
            self.b.append(arrays[2*i+1])
    def forward(self, X):
        A = X
        for i in range(len(self.W)-1):
            A = relu(A @ self.W[i] + self.b[i])
        ZL = A @ self.W[-1] + self.b[-1]
        return softmax(ZL)
    def predict(self, X):
        probs = self.forward(X)
        return probs.argmax(axis=1), probs

# ---------- Preprocesamiento ----------
def preprocess_roi(roi_gray, invert_auto=True, debug=False):
    blur = cv2.GaussianBlur(roi_gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert_auto:
        if (th.mean() / 255.0) > 0.5:
            th = 255 - th
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        img28 = np.zeros((28,28), dtype=np.uint8)
        return img28.reshape(1, 784).astype(np.float32)/255.0
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    digit = th[y:y+h, x:x+w]
    if w > h:
        new_w = 20; new_h = max(1, int(h * (20.0 / w)))
    else:
        new_h = 20; new_w = max(1, int(w * (20.0 / h)))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28,28), dtype=np.uint8)
    x_off = (28 - new_w)//2; y_off = (28 - new_h)//2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit_resized
    if debug:
        cv2.imshow("th", th); cv2.imshow("digit_resized", digit_resized)
    x = canvas.astype(np.float32) / 255.0
    return x.reshape(1, 784)

# ---------- Cámara robusta ----------
def try_open_any_camera(max_index=5, width=1280, height=720):
    for idx in range(max_index+1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # Intenta setear tamaño (no todos los backends respetan esto)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Prueba lectura
            ok, _ = cap.read()
            if ok:
                print(f"[OK] Cámara abierta en índice {idx}")
                return cap, idx
            cap.release()
    return None, None

def main():
    print("Cargando pesos desde weights_mlp.npz ...")
    model = MLPInference("weights_mlp.npz", layer_sizes=(784,128,64,10))

    cap, cam_idx = try_open_any_camera(max_index=5)
    if cap is None:
        print("No pude abrir ninguna cámara. Revisa permisos en macOS (Privacy → Camera) y vuelve a intentar.")
        return

    invert_manual = False
    auto_invert = True
    roi_size = 300
    print("Controles: q=salir | i=toggle invert manual | a=toggle auto-invert | c=cambiar cámara")

    current_idx = cam_idx
    last_switch = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            # Si falla la lectura, intenta reconectar a la misma cámara
            cap.release()
            time.sleep(0.2)
            cap = cv2.VideoCapture(current_idx)
            ok, frame = cap.read()
            if not ok:
                cv2.waitKey(1)
                continue

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        cx, cy = W//2, H//2
        x1, y1 = cx - roi_size//2, cy - roi_size//2
        x2, y2 = cx + roi_size//2, cy + roi_size//2
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        roi = frame[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        x = preprocess_roi(roi_gray, invert_auto=auto_invert and not invert_manual)

        pred, probs = model.predict(x)
        pred = int(pred[0]); conf = float(probs[0, pred])

        # Overlay de texto
        txt1 = f"Pred: {pred}  conf: {conf*100:.1f}%"
        txt2 = f"invert: {'AUTO' if auto_invert else ('MANUAL ON' if invert_manual else 'OFF')} | cam:{current_idx}"
        cv2.putText(frame, txt1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,200,255), 2)
        cv2.putText(frame, txt2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        cv2.putText(frame, "Escribe el digito dentro del cuadro verde", (20, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

        # Vista del 28x28
        vis28 = (x.reshape(28,28) * 255).astype(np.uint8)
        vis28 = cv2.resize(vis28, (140,140), interpolation=cv2.INTER_NEAREST)
        frame[10:10+140, W-10-140:W-10] = cv2.cvtColor(vis28, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, (W-10-140, 10), (W-10, 150), (255,255,255), 1)

        cv2.imshow("MNIST Webcam Demo", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('i'):
            invert_manual = not invert_manual
        elif key == ord('a'):
            auto_invert = not auto_invert
        elif key == ord('c'):
            # Cambiar de cámara al vuelo
            now = time.time()
            if now - last_switch > 0.3:  # anti-rebote
                last_switch = now
                next_idx = (current_idx + 1) % 6
                cap.release()
                cap = cv2.VideoCapture(next_idx)
                ok, _ = cap.read()
                if ok:
                    current_idx = next_idx
                    print(f"[Switch] Cámara -> índice {current_idx}")
                else:
                    # si no sirve, volver
                    cap.release()
                    cap = cv2.VideoCapture(current_idx)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
