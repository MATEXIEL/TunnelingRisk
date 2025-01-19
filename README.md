# TunnelingRiskPrediction

Bu proje, tünel kazı projeleri için risk değerlendirmesi yapmayı amaçlayan bir makine öğrenimi uygulamasını kapsamaktadır. Kullanılan veri seti, tünel kazılarının çeşitli mühendislik parametrelerini, jeolojik faktörleri ve araştırma verilerini içermektedir. Hedef değişkeni olan "Risk Seviyesi", Düşük Risk (0) ile Kritik Risk (3) arasında değerler almaktadır.

---

# İçerik
Proje hakkında kısa bir tanıtım videosu: https://youtu.be/MVyyxX92l4Y

---

# Kod Tanıtımı
- **1. Veri Yükleme**
  - Veri seti pandas kütüphanesi yardımıyla bir CSV dosyasından yüklenmektedir:
  - ```python
    file_path = '/content/tunneling_risk.csv' # Dosya adını doğru girdiğinizden emin olun
    data = pd.read_csv(file_path)
    ```
    
- **2. Veri Temizleme**
  - Gereksiz Kolonların Çıkarılması
  - Tünel ID'leri gibi analiz için bir anlam ifade etmeyen kolonlar veri setinden kaldırılır
  - ```python
    data = data.iloc[:, 1:]
    ```
  - Eksik Verilerin Ele Alınması
  - Eksik değerlerin bulunduğu satırlar tespit edilerek silinir
  - ```python
    data = data.dropna()
    ```

- **3. Kategorik Verilerin Dönüştürülmesi**
  - Sıklıkla Kullanılan Kategorilerin Gruplanması
  - Nadir kategoriler "Other" olarak yeniden adlandırılır
  - ```python
    threshold = 10
    threshold = 10
    for col in categorical_columns:
        if data[col].nunique() > threshold:
            top_categories = data[col].value_counts().nlargest(threshold).index
            data[col] = data[col].apply(lambda x: x if x in top_categories else 'Other')
    ```
  - *One-Hot Encoding*
  - Kategorik veriler **binary** formatta kodlanır
  - ```python
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    ```

- **4. Korelasyon Analizi**
  - Risk Seviyesi ile diğer feature'lar arasındaki ilişkiler korelasyon matrisiyle analiz edilmiştir
  - ```python
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    ```
    
- **5. Model Eğitimi ve Değerlendirme**    
  - Kullanılan Modeller:
    - Random Forest
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
  - Modeller eğitilir ve test veri setindeki başarıları karşılaştırılır. En yüksek başarıyı sağlayan model "en iyi model" olarak seçilmiştir.
  - ```python
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
    ```

- **6. Sonuçların Değerlendirilmesi**
  - En iyi modelin performansı:
  - ```python
    print(classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    ```

---

# Sonuç ve Analiz
Bu proje, tünel kazı risklerini değerlendirmek için etkili bir makine öğrenimi pipeline'ı sunmaktadır. Model performansının %99 doğruluk oranına uğraşması, verilerin çok iyi temsili ve önceden temizlenmiş yapısı sayesinde mümkün olmuştur.
  **Bu Çalışmanın Önemi**
  - Gerçek zamanlı karar alma sistemlerine altyapı oluşturabilir.
  - Proje maliyetlerini ve tünel kazı esnasında ortaya çıkabilecek riskleri azaltabilir.
  - Su altı ulaşım çözümlerinde daha modüler ilerlenmesine olanak sağlayabilir. 

  **Gelecek Çalışmalar**
  - Daha fazla veri ile eğitim yapılarak model genelleştirilebilir.
  - Risk seviyesi ile zamansal ve mekansal faktörler arasındaki farklı bağlantılar incelenebilir.
  - Gelişmiş derin öğrenme modelleriyle çalışma genişletilebilir.

  **Model Seçimi ve Performans Analizi**
  - Random Forest modeli, bu veri setine en iyi uyum sağlayan model olarak göze çarptı. Modelin ağırlıklarının optimize edilmesi ve hiperparametre ayarlamalarıyla daha uygun bir performans sergilemesi sağlandı. Elde edilen bu doğruluk oranı, gerçek hayatla paralel veya aynı veriler üzerinde çalışıldığı müddetçe neredeyse kesin bir risk analizi yapılabileceğini gösterir. 
