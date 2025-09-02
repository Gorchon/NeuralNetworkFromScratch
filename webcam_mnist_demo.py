import cv2
import numpy as np

# ---------- MLP inference-only (carga pesos .npz guardados) ----------

def relu(x): return np.maximum(0.0, x)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

class MLPInference:
    def __init__(self, weight_path="weights_mlp.npz", layer_sizes=(784,128,64,10)):
        self.W = []
        self.b = []
        data = np.load(weight_path, allow_pickle=False)
        arrays = [data[k] for k in data.files]
        L = len(layer_sizes) - 1
        for i in range(L):
            self.W.append(arrays[2*i])     # (fan_in, fan_out)
            self.b.append(arrays[2*i+1])   # (1, fan_out)

    def forward(self, X):
        A = X
        # Todas menos la última con ReLU
        for i in range(len(self.W) - 1):
            A = relu(A @ self.W[i] + self.b[i])
        ZL = A @ self.W[-1] + self.b[-1]
        return softmax(ZL)

    def predict(self, X):
        probs = self.forward(X)
        return probs.argmax(axis=1), probs

# ---------- Preprocesamiento del ROI a "formato MNIST" ----------

def preprocess_roi(roi_gray, invert_auto=True, debug=False):
    """
    Toma un ROI en escala de grises (uint8), encuentra el dígito y
    genera un vector 1x784 normalizado en [0,1] similar a MNIST.
    """
    # 1) Suavizar y binarizar (Otsu)
    blur = cv2.GaussianBlur(roi_gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2) A veces fondo/dígito vienen invertidos. Decidir si invertir.
    if invert_auto:
        # Si la imagen binaria tiene más "blanco" que "negro", asumimos que el fondo está blanco
        # y el dígito negro; invertimos para que dígito sea blanco sobre fondo negro (como MNIST).
        white_ratio = th.mean() / 255.0
        # Si hay mucho blanco, interpretamos que fondo es blanco => invertimos.
        if white_ratio > 0.5:
            th = 255 - th
    else:
        # Mantener como está
        pass

    # 3) Encontrar contorno principal (dígito)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        # Si no hay contornos, devolvemos un "cero" plano para evitar crashes
        img28 = np.zeros((28,28), dtype=np.uint8)
        return img28.reshape(1, 784).astype(np.float32)/255.0

    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    digit = th[y:y+h, x:x+w]

    # 4) Normalizar aspecto: redimensionar el digit manteniendo ratio a un box 20x20 y centrar en 28x28
    # Escala al tamaño que quepa en 20 manteniendo proporción
    if w > h:
        new_w = 20
        new_h = int(h * (20.0 / w))
        if new_h == 0: new_h = 1
    else:
        new_h = 20
        new_w = int(w * (20.0 / h))
        if new_w == 0: new_w = 1

    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5) Pegar en lienzo 28x28 centrado
    canvas = np.zeros((28,28), dtype=np.uint8)
    x_offset = (28 - new_w)//2
    y_offset = (28 - new_h)//2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    if debug:
        cv2.imshow("th", th)
        cv2.imshow("digit_resized", digit_resized)

    # 6) Normalizar a [0,1] y "aplanar"
    x = canvas.astype(np.float32) / 255.0
    return x.reshape(1, 784)

# ---------- Demo de cámara ----------

def main():
    print("Cargando pesos desde weights_mlp.npz ...")
    model = MLPInference("weights_mlp.npz", layer_sizes=(784,128,64,10))

    cap = cv2.VideoCapture(0)  # puede ser 0, 1, etc. según tu sistema
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Revisa permisos o índice de cámara.")
        return

    invert_manual = False   # tecla 'i' para alternar inversión manual
    auto_invert = True      # por defecto auto
    roi_size = 280          # tamaño del recuadro ROI en pantalla
    print("Controles: q=salir, i=toggle invert, a=toggle auto-invert")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear horizontal (opcional, tipo espejo)
        frame = cv2.flip(frame, 1)

        H, W = frame.shape[:2]
        cx, cy = W//2, H//2
        x1, y1 = cx - roi_size//2, cy - roi_size//2
        x2, y2 = cx + roi_size//2, cy + roi_size//2

        # Asegurar límites
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)

        # Dibujar ROI
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        roi = frame[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Preprocesar ROI -> 1x784
        x = preprocess_roi(roi_gray, invert_auto=auto_invert and not invert_manual)

        # Predicción
        pred, probs = model.predict(x)
        pred = int(pred[0])
        conf = float(probs[0, pred])

        # Mostrar información
        txt = f"Pred: {pred}  conf: {conf*100:.1f}%  inv:{'auto' if auto_invert else ('manual ON' if invert_manual else 'off')}"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 255), 2)

        # Mostrar también el 28x28 (escalar para visualizar)
        vis28 = (x.reshape(28,28) * 255).astype(np.uint8)
        vis28 = cv2.resize(vis28, (140,140), interpolation=cv2.INTER_NEAREST)
        frame[10:10+140, W-10-140:W-10] = cv2.cvtColor(vis28, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, (W-10-140, 10), (W-10, 10+140), (255,255,255), 1)

        cv2.imshow("MNIST Webcam Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            # Forzar inversión manual (conmuta)
            invert_manual = not invert_manual
        elif key == ord('a'):
            # Activar/desactivar auto-invert
            auto_invert = not auto_invert

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
