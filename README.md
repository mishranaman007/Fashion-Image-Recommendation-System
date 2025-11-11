# Fashion-Image-Recommendation-System
Deep learning–based fashion recommendation system using ResNet50 and cosine similarity to suggest visually similar clothing items.



# 👗 Fashion Image Recommendation System

### 🧠 Overview

The **Fashion Image Recommendation System** is an intelligent, deep-learning–based project that identifies and recommends **visually similar fashion products** (such as shirts, dresses, shoes, and accessories) from a large image dataset.

By leveraging **Computer Vision** and **Deep Feature Extraction** through a pre-trained **ResNet50** Convolutional Neural Network (CNN), this system understands image patterns — including **texture, color, and shape** — and finds the top 5 similar items for any user-uploaded fashion image.

This project demonstrates practical use of **content-based image retrieval (CBIR)** and **recommendation systems**, combining the power of **AI and fashion technology**.

---

## 🎯 Key Objectives

* To build an **AI-powered recommendation system** that analyzes fashion images visually.
* To extract **high-level features** from images using **ResNet50 (Transfer Learning)**.
* To implement **cosine similarity** for finding similar fashion items.
* To create an **interactive Colab application** where users can upload any clothing image and get 5 visually similar recommendations.

---

## 🧠 Core Concept

Unlike text-based recommendation engines that depend on descriptions or tags, this system is **purely image-driven**.
It understands *what an image looks like*, not *what it’s called*.

For example:
If a user uploads an image of a **red floral dress**, the model will analyze its visual pattern and recommend **other dresses with similar colors, textures, and styles**, even if their filenames or labels differ.

---

## ⚙️ Project Workflow

### 🔹 Step 1 — Dataset Preparation

* The dataset consists of ~44,000 fashion images organized into folders.
* The images are extracted and preprocessed using `tensorflow.keras.preprocessing`.

### 🔹 Step 2 — Feature Extraction (ResNet50)

* A pre-trained **ResNet50** model (trained on ImageNet) is used as a **feature extractor**.
* For each image, a **2048-dimensional feature vector** is generated, representing its visual characteristics.
* These embeddings are stored in NumPy arrays for efficient similarity search.

### 🔹 Step 3 — Similarity Computation

* When a user uploads a new image, its features are extracted using the same ResNet50 model.
* **Cosine similarity** measures how close this new image is to every image in the dataset.
* The top 5 most similar images are returned as recommendations.

### 🔹 Step 4 — Visualization

* The uploaded image and its 5 most similar fashion items are displayed side-by-side for visual comparison.
* All processing is done interactively in **Google Colab** for easy demonstration.

---

## 💻 Tech Stack

| Category                 | Technology                                                      |
| ------------------------ | --------------------------------------------------------------- |
| **Language**             | Python                                                          |
| **Framework / Platform** | Google Colab                                                    |
| **Deep Learning Model**  | ResNet50 (from TensorFlow / Keras)                              |
| **Libraries Used**       | TensorFlow, NumPy, scikit-learn, Matplotlib, Pillow (PIL), tqdm |
| **Algorithm**            | Cosine Similarity for image feature comparison                  |
| **Storage Format**       | NumPy `.npy` arrays (for extracted features)                    |

---

## 🚀 Features

✅ Accepts user-uploaded fashion images (JPG, JPEG, PNG)
✅ Extracts deep visual features using ResNet50 (Transfer Learning)
✅ Finds top 5 visually similar items using cosine similarity
✅ Displays recommendations instantly in Colab
✅ Can process large datasets (40k+ images)
✅ Fully modular and extendable (for web app or API integration)

---

## 🧩 Project Structure

```
Fashion-Image-Recommendation-System/
│
├── fashion_recommender.ipynb     # Main project notebook
├── fashion_dataset/              # Extracted dataset (optional)
├── features.npy                  # Saved deep feature vectors
├── image_paths.npy               # Image paths corresponding to features
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation (this file)
```

---

## 📸 Example Output

**Input:**
A user uploads an image of a **blue denim jacket**

**Output:**
The system recommends 5 similar jackets from the dataset with matching style, texture, and color.

```plaintext
🖼 Uploaded Image → 👕 Recommended Similar Images
```

*(Include a screenshot of your output here — name it `demo.png` and add below)*

```markdown
![Demo Screenshot](demo.png)
```

---

## 🔍 How to Run the Project

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload your dataset ZIP file (e.g., `fashion_images.zip`).
3. Run each cell in the notebook sequentially.
4. When prompted, upload any test fashion image (e.g., `.jpeg`, `.jpg`, `.png`).
5. View the top 5 visually similar recommendations.

---

## 🏁 Results

✅ Successfully implemented a **Content-Based Image Retrieval** system for fashion products.
✅ Achieved fast and accurate visual similarity detection using **ResNet50** features.
✅ Demonstrated how deep learning can bridge **AI and fashion e-commerce**.

---

## 🚀 Future Enhancements

* ⚡ Integrate **FAISS** (Facebook AI Similarity Search) for faster image retrieval.
* 🌐 Develop a **Streamlit or Flask web interface** for public use.
* 📱 Add category-wise filtering (e.g., only shirts, only shoes).
* 🧠 Fine-tune ResNet50 on custom fashion datasets for better domain performance.
* 🧩 Deploy as a web service or plug into an e-commerce recommendation engine.

---

## 📦 Dependencies

Create a `requirements.txt` file with:

```
tensorflow
numpy
scikit-learn
matplotlib
pillow
tqdm
```

Then install using:

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

## **Naman Mishra** <br>
🎓 Computer Science Student <br>
📧 [mishranaman80773@gmail.com](mailto:mishranaman80773@gmail.com) <br>
💼 Passionate about exploring immersive technologies i.e AI and AR to solve real-world problems 

## **Pratyush Mukherjee** <br>
🎓 Computer Science Student <br>
📧 [pratyushmukherjee_202210101150058@srmu.ac.in@srmu.ac.in](mailto:pratyushmukherjee_202210101150058@srmu.ac.in@srmu.ac.in) <br>
💼 Passionate about applying AI to solve real-world problems like fashion, healthcare, and automation. 

## **Kritagya Bhagat** <br>
🎓 Computer Science Student <br>
📧 [kritagyabhagat_202210101150036@srmu.ac.in](mailto:kritagyabhagat_202210101150036@srmu.ac.in) <br>
💼 Passionate about exploring Java Technology 

---

## 🏆 Key Takeaways

* Implemented **Deep Learning for Image Similarity**
* Used **Transfer Learning** effectively with **ResNet50**
* Created a **real-world fashion recommendation system** from scratch
* Demonstrated **AI application in fashion tech and visual search**

---

