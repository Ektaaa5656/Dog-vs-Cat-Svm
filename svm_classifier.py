import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import Counter

# 👇 Dataset path
DATASET_PATH = "train"
IMG_SIZE = 64  # Resize to smaller size for speed

def extract_hog_features(img):
    # HOG = shape, texture, edge information
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False, channel_axis=-1)
    return features

def load_data(path):
    X, y = [], []

    cat_files = [f for f in os.listdir(path) if f.startswith("cat")][:1000]
    dog_files = [f for f in os.listdir(path) if f.startswith("dog")][:1000]
    files = cat_files + dog_files

    for img_name in tqdm(files, desc="📸 Extracting HOG features"):
        label = 1 if 'dog' in img_name else 0
        img_path = os.path.join(path, img_name)
        
        # 🖼️ Read color image & resize
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # ✨ HOG features from color image
        features = extract_hog_features(img)
        X.append(features)
        y.append(label)

    print("📊 Label Counts:", Counter(y))
    return np.array(X), np.array(y)

print("🔄 Loading and processing images...")
X, y = load_data(DATASET_PATH)

print("🔁 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training SVM with RBF kernel...")
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

print("✅ Predicting...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"🎯 Improved Accuracy: {acc * 100:.2f}%")


# 🧪 Predict single image...

def predict_single_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Image not found at {image_path}")
        return
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    hog_feat = extract_hog_features(img_resized)
    prediction = model.predict([hog_feat])[0]
    label = "🐶 Dog" if prediction == 1 else "🐱 Cat"

    # 🖼️ Show image with predicted label
    cv2.putText(img, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 🔍 Test with a known image

# test_image_path = "train/<correct-image-name>.jpg"....

# 🔍 Test with a known cat image

test_image_path = "train/cat.6.jpg"  # <-- Change this to an actual cat image name
predict_single_image(test_image_path)



#  for testing the image run this line in new terminal ( python svm_classifier.py )..
