# smart-irrigation-ml-esp32
ESP32 smart irrigation with water prediction using ML
# 🌱 Smart Irrigation System using Machine Learning – ESP32

Système d’arrosage intelligent basé sur **la prédiction de la quantité d’eau nécessaire** à partir de capteurs environnementaux et d’un modèle de Machine Learning embarqué sur **ESP32 WROOM**.

---

## 🎯 Objectif du projet
- Prédire automatiquement la quantité d’eau nécessaire pour l’irrigation
- Réduire la consommation d’eau
- Automatiser l’arrosage
- Appliquer le Machine Learning sur microcontrôleur

---

## 📊 Données d’entrée (capteurs)
- 🌡️ Température – DHT22  
- 🌱 Humidité du sol – Water Sensor  
- ☀️ Luminosité – LDR  
- ⏱️ Temps entre deux arrosages – Horloge ESP32  

---

## 🧠 Intelligence Artificielle
- Dataset : Fichier Excel réel
- Entraînement en Python
- Modèle exporté en **TensorFlow Lite (.tflite)**
- Intégration directe dans l’ESP32

---

## ⚙️ Fonctionnement
1. Lecture des capteurs
2. Prétraitement des données
3. Prédiction de la quantité d’eau
4. Activation de la pompe via relais
5. Affichage sur le moniteur série

---

## 🛠️ Matériel utilisé
- ESP32 WROOM
- DHT22
- Water Sensor
- LDR
- Module relais
- Pompe à eau
- Alimentation externe

---

## 💻 Technologies
- Arduino / C++
- Python (Machine Learning)
- TensorFlow Lite
- IoT & systèmes embarqués

---

## 📁 Structure du projet
- `dataset/` → Données Excel
- `ml_model/` → Modèle ML + TFLite
- `esp32_code/` → Code Arduino
- `schematic/` → Schémas

---

## 🚀 Perspectives
- Application mobile
- Prédiction multi-plantes
- Historique des données
- Notifications intelligentes

---

## 👤 Auteur
- Nom : [ajana abdelali]
- Formation : [Engineering student | Embedded electronics systems & controls]
- Pays : Maroc
- LinkedIn : [www.linkedin.com/in/abdelali-ajana-27559127]
