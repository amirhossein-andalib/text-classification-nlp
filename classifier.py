from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import numpy as np


class TextClassifier:
    def __init__(self, model_type='svm'):
        self.model_type = model_type
        self.pipeline = self._build_pipeline()
        self.classes = None

    def _build_pipeline(self):
        """Build sklearn pipeline with TF-IDF and classifier"""

        # Choose classifier
        if self.model_type == 'svm':
            classifier = LinearSVC(max_iter=1000)
        elif self.model_type == 'naive_bayes':
            classifier = MultinomialNB()
        elif self.model_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000)
        else:
            classifier = LinearSVC(max_iter=1000)

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )),
            ('classifier', classifier)
        ])

        return pipeline

    def train(self, X_train, y_train):
        """Train the classifier"""
        self.pipeline.fit(X_train, y_train)
        self.classes = self.pipeline.classes_
        print(f"✅ مدل {self.model_type.upper()} با موفقیت آموزش دید")
        # تبدیل به str معمولی، وگرنه با numpy 2.x چیزی شبیه
        # np.str_('sports') به‌جای 'sports' چاپ می‌شد
        print(f"   کلاس‌ها: {[str(c) for c in self.classes]}")

    def evaluate(self, X_test, y_test):
        """Evaluate the classifier"""
        y_pred = self.pipeline.predict(X_test)

        print("\n📊 گزارش ارزیابی:")
        print("=" * 55)
        # zero_division=0: قبلاً وقتی یک کلاس هیچ پیش‌بینی‌ای نمی‌گرفت،
        # یک UndefinedMetricWarning زشت توی کنسول چاپ می‌شد؛ این پارامتر
        # آن هشدار را خاموش می‌کند و دقیقاً ۰ را به‌جای آن می‌گذارد.
        print(classification_report(y_test, y_pred, zero_division=0))

        print("📊 ماتریس درهم‌ریختگی:")
        print("=" * 55)
        # باگ مهم: قبلاً classes فقط از روی y_test ساخته می‌شد
        # (sorted(set(y_test)))، در حالی که confusion_matrix به‌طور
        # پیش‌فرض از union برچسب‌های y_test و y_pred استفاده می‌کند. اگر
        # حتی یک کلاس فقط در y_pred ظاهر شود (نه در y_test)، اندازه‌ی
        # ماتریس با تعداد برچسب‌های چاپ‌شده ناهم‌خوان می‌شد. الان با
        # پارامتر labels=classes، هر دو را همگام و هم‌ترتیب می‌کنیم.
        classes = sorted(set(y_test) | set(y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        print(f"{'':12}", end='')
        for c in classes:
            print(f"{c:12}", end='')
        print()
        for i, row in enumerate(cm):
            print(f"{classes[i]:12}", end='')
            for val in row:
                print(f"{val:<12}", end='')
            print()

        accuracy = np.mean(y_pred == y_test)
        print(f"\n✅ دقت کلی: {accuracy * 100:.2f}%")
        return accuracy

    def predict(self, text):
        """Predict category of a single text"""
        prediction = self.pipeline.predict([text])[0]
        return prediction

    def predict_batch(self, texts):
        """Predict categories for multiple texts"""
        return self.pipeline.predict(texts)
