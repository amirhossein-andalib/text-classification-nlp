# Persian Text Classification with TF-IDF

Multi-class Persian text classifier using TF-IDF features
and machine learning algorithms.

## Features
- Persian text preprocessing with Hazm
- TF-IDF vectorization with bigrams
- Three classifier comparison:
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Logistic Regression
- Detailed evaluation with classification report
- Confusion matrix visualization

## Categories
- ورزشی / Sports
- سیاسی / Politics
- علمی / Science
- فرهنگی / Culture

## Dataset
۶۰ نمونه (۱۵ در هر دسته). نسخه‌ی اولیه فقط ۲۴ نمونه (۶ در هر دسته)
داشت که باعث می‌شد TF-IDF با بای‌گرام بیش‌برازش (overfit) کند و دقت
روی داده‌ی آزمون فقط ۱۷-۳۳٪ باشد؛ با افزایش داده به دقت ~۸۰٪ رسید.

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python main.py
```

## Technologies
- Python 3.x
- Hazm — Persian NLP
- Scikit-learn — Machine Learning
- NumPy

## Author
Seyed Amirhossein Andalib — Linguistics & NLP
