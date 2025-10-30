# 🏥 Medical Cost Prediction

مشروع لتوقع تكلفة التأمين الطبي باستخدام بيانات تجريبية وموديل Random Forest.

## هيكل المشروع
```
medical_cost_prediction/
├── data/                  # لا يتضمن الملف محليًا — يتم تحميله من GitHub عند التشغيل
├── models/                # سيحفظ فيه النموذج (best_model.pkl)
├── notebooks/             # يحتوي على الـ Jupyter Notebook (تحليل + تدريب)
├── app.py                 # واجهة Gradio لتجربة النموذج
├── train_model.py         # سكريبت تدريب النموذج وحفظه
├── requirements.txt       # المتطلبات
└── README.md
```

## تشغيل المشروع
1. تثبيت المتطلبات:
   ```bash
   pip install -r requirements.txt
   ```
2. لتدريب النموذج وحفظه:
   ```bash
   python train_model.py
   ```
3. لتشغيل الواجهة:
   ```bash
   python app.py
   ```

## ملاحظات
- ملف البيانات يتم تحميله تلقائيًا من GitHub ضمن السكريبت والـNotebook.
- الـNotebook يشرح الخطوات بالشرح العربي المبسط ويحتوي رسومًا بيانية توضيحية.
