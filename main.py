from data_preparation import DataPreparation, get_sample_data
from classifier import TextClassifier
from sklearn.model_selection import train_test_split


def main():
    print("=" * 55)
    print("   PERSIAN TEXT CLASSIFICATION")
    print("=" * 55)

    # ── Step 1: Load Data ──
    print("\n📂 بارگذاری داده‌ها...")
    texts, labels = get_sample_data()
    print(f"   تعداد کل متون: {len(texts)}")
    print(f"   دسته‌بندی‌ها: {set(labels)}")

    # ── Step 2: Preprocess ──
    print("\n⚙️  پیش‌پردازش متون...")
    prep = DataPreparation()
    processed_texts, labels = prep.prepare_dataset(texts, labels)
    print("   پیش‌پردازش انجام شد ✅")

    # ── Step 3: Split Data ──
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )
    print(f"\n📊 تقسیم داده:")
    print(f"   آموزش: {len(X_train)} نمونه")
    print(f"   آزمون: {len(X_test)} نمونه")

    # ── Step 4: Train & Evaluate Models ──
    models = ['svm', 'naive_bayes', 'logistic']

    best_accuracy = 0
    best_model = None

    for model_type in models:
        print(f"\n{'=' * 55}")
        print(f"🤖 مدل: {model_type.upper()}")
        print('=' * 55)

        classifier = TextClassifier(model_type=model_type)
        classifier.train(X_train, y_train)
        accuracy = classifier.evaluate(X_test, y_test)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = classifier

    # ── Step 5: Test with New Texts ──
    print("\n" + "=" * 55)
    print("🔍 تست با متون جدید:")
    print("=" * 55)

    new_texts = [
        "بازی فوتبال امشب هیجان‌انگیز بود",
        "دانشمندان کشف جدیدی در علم پزشکی داشتند",
        "نمایشگاه هنری در گالری شهر برگزار شد",
        "وزیر اقتصاد برنامه جدیدی اعلام کرد",
    ]

    for text in new_texts:
        processed = prep.preprocess(text)
        prediction = best_model.predict(processed)

        label_map = {
            'sports': 'ورزشی 🏅',
            'politics': 'سیاسی 🏛️',
            'science': 'علمی 🔬',
            'culture': 'فرهنگی 🎭'
        }

        print(f"متن      : {text}")
        print(f"دسته‌بندی : {label_map.get(prediction, prediction)}")
        print("-" * 55)


if __name__ == "__main__":
    main()
