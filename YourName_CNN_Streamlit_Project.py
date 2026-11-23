import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# PART A: CNN IMAGE CLASSIFICATION (MNIST)
# ==========================================
def run_cnn_part():
    st.header("Part A: CNN Image Classification (MNIST)")
    st.write("Dataset: Handwritten Digits (0–9) | Image Shape: 28x28 Grayscale")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    st.write(f"**Training Samples:** {x_train.shape} | **Test Samples:** {x_test.shape}")

    st.subheader("1. Sample Dataset Images")
    fig_samples, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(x_train[i], cmap="gray")
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis("off")
    st.pyplot(fig_samples)

    x_train_norm = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test_norm = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    st.subheader("2. CNN Model Summary")
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    st.code("\n".join(summary_lines))

    epochs = st.slider("Select Training Epochs", min_value=2, max_value=10, value=5)
    if st.button("Train & Evaluate CNN Model"):
        with st.spinner("Training CNN model... please wait"):
            history = model.fit(x_train_norm, y_train, epochs=epochs, batch_size=64, validation_split=0.1, verbose=0)

        test_loss, test_acc = model.evaluate(x_test_norm, y_test, verbose=0)
        st.success(f"Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

        st.subheader("3. Training & Validation Curves")
        fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["accuracy"], label="Train Accuracy")
        ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
        ax1.set_title("Training & Validation Accuracy")
        ax1.legend()

        ax2.plot(history.history["loss"], label="Train Loss")
        ax2.plot(history.history["val_loss"], label="Val Loss")
        ax2.set_title("Training & Validation Loss")
        ax2.legend()
        st.pyplot(fig_hist)

        st.subheader("4. Sample Predictions (Correct vs. Incorrect)")
        predictions = np.argmax(model.predict(x_test_norm, verbose=0), axis=1)
        correct_idx = np.where(predictions == y_test)[:3]
        incorrect_idx = np.where(predictions != y_test)[:3]
        sample_idx = np.concatenate([correct_idx, incorrect_idx])

        fig_pred, axes_p = plt.subplots(1, len(sample_idx), figsize=(12, 2.5))
        for i, idx in enumerate(sample_idx):
            axes_p[i].imshow(x_test[idx], cmap="gray")
            status = "Correct" if predictions[idx] == y_test[idx] else "Wrong"
            axes_p[i].set_title(f"{status}\nAct:{y_test[idx]} | Pred:{predictions[idx]}")
            axes_p[i].axis("off")
        st.pyplot(fig_pred)

# ==========================================
# PART B: STREAMLIT APPLICATION
# ==========================================
def run_streamlit_app():
    st.header("Part B: Student Marks & Grade Calculator")
    st.write("Enter student details and marks below to calculate percentage, grade, and pass/fail status.")

    student_name = st.text_input("Student Name", placeholder="e.g., Alex Johnson")

    col1, col2, col3 = st.columns(3)
    with col1:
        math_marks = st.number_input("Mathematics (0–100)", min_value=-10.0, max_value=120.0, value=75.0)
    with col2:
        science_marks = st.number_input("Science (0–100)", min_value=-10.0, max_value=120.0, value=80.0)
    with col3:
        english_marks = st.number_input("English (0–100)", min_value=-10.0, max_value=120.0, value=70.0)

    if st.button("Calculate Result"):
        if not student_name.strip():
            st.error("Invalid Input: Please enter the student's name.")
            return

        marks = [math_marks, science_marks, english_marks]
        if any(m < 0 or m > 100 for m in marks):
            st.warning("Invalid Input: All marks must be strictly between 0 and 100.")
            return

        total = sum(marks)
        percentage = (total / 300.0) * 100.0

        if percentage >= 90:
            grade = "A+"
        elif percentage >= 75:
            grade = "A"
        elif percentage >= 60:
            grade = "B"
        elif percentage >= 50:
            grade = "C"
        else:
            grade = "F"

        status = "PASS" if all(m >= 40 for m in marks) else "FAIL (Minimum 40 required per subject)"

        st.subheader(f"Result Summary for {student_name}")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Total Marks", f"{total:.1f} / 300")
        res_col2.metric("Percentage", f"{percentage:.2f}%")
        res_col3.metric("Grade", grade)

        if "PASS" in status:
            st.success(f"Status: {status}")
        else:
            st.error(f"Status: {status}")

# ==========================================
# MAIN APP NAVIGATION
# ==========================================
def main():
    st.set_page_config(page_title="CNN & Streamlit Minor Project", layout="wide")
    st.sidebar.title("Project Navigation")
    choice = st.sidebar.radio("Select Section:", ["Part B: Streamlit App", "Part A: CNN Classification"])

    if choice == "Part B: Streamlit App":
        run_streamlit_app()
    else:
        run_cnn_part()

if __name__ == "__main__":
    main()
