# Dog-vs-Cat-Svm
In this project, we developed a machine learning model to classify images of dogs and cats using a Support Vector Machine (SVM). The goal was to train a model that can distinguish between the two classes based on image features.

We first preprocessed the dataset by resizing the images to a uniform size and converting them to grayscale to reduce computational complexity. Feature extraction was performed using Histogram of Oriented Gradients (HOG), which captures edge and texture information essential for distinguishing animals.

After feature extraction, we split the dataset into training and testing sets and trained an SVM classifier with a linear kernel. The model was evaluated using accuracy metrics on the test set. The SVM demonstrated strong performance in correctly classifying unseen dog and cat images.

This project demonstrates how classical machine learning techniques like SVM, combined with effective feature engineering, can be applied to computer vision problems.
