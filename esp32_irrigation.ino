#include "modele_irrigation.h"
#include <Arduino.h>

// ===== TensorFlow Lite Micro =====
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===== CAPTEURS =====
#include <DHT.h>

// ------------------ ML VARIABLES ------------------
const int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// ------------------ CAPTEURS ------------------
#define DHT_PIN 4
#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

#define LDR_PIN 34
#define PH_PIN 35
#define HUMIDITE_SOL_PIN 32

// ------------------ LED BUILTIN ------------------
#define LED_BUILTIN 2

// ------------------ POMPE ------------------
#define PUMP_PIN 14

void setupPump() {
    pinMode(PUMP_PIN, OUTPUT);
    digitalWrite(PUMP_PIN, LOW);
}

void pompeON() {
    digitalWrite(PUMP_PIN, HIGH);
}

void pompeOFF() {
    digitalWrite(PUMP_PIN, LOW);
}

// ------------------ LECTURE CAPTEURS ------------------
float lireTemperature() {
    float t = dht.readTemperature();
    return isnan(t) ? 25.0 : t;
}

float lireHumiditeSol() {
    int raw = analogRead(HUMIDITE_SOL_PIN);
    // Inverser la lecture : plus la valeur est basse, plus le sol est humide
    float pourcentage = map(raw, 0, 4095, 100, 0);
    return constrain(pourcentage, 0, 100);
}

float lirePH() {
    int raw = analogRead(PH_PIN);
    float voltage = (raw / 4095.0) * 3.3;
    float ph = 7.0 + ((2.5 - voltage) / 0.18);
    return constrain(ph, 0, 14);
}

float lireLuminosite() {
    int raw = analogRead(LDR_PIN);
    return map(raw, 0, 4095, 0, 2000);
}

// ------------------ INITIALISATION ML ------------------
void initMLModel() {
    Serial.println("🔄 Initialisation modèle ML...");
    
    model = tflite::GetModel(model_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.print("❌ Version modèle : ");
        Serial.print(model->version());
        Serial.print(", attendu : ");
        Serial.println(TFLITE_SCHEMA_VERSION);
        while(1) {
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
            delay(500);
        }
    }

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("❌ Erreur allocation TFLM");
        while(1) {
            digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
            delay(300);
        }
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.print("✔️ Modèle ML prêt ! Input dims: ");
    Serial.println(input_tensor->dims->data[1]);
}

// ------------------ SETUP ------------------
unsigned long dernier_arrosage = 0;

void setup() {
    // Configuration LED built-in
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);
    
    // Initialisation Serial
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("\n=== Initialisation système irrigation ===");
    Serial.flush();

    // Initialisation capteurs
    Serial.println("🔄 Initialisation capteurs...");
    dht.begin();
    delay(1000);

    pinMode(LDR_PIN, INPUT);
    pinMode(PH_PIN, INPUT);
    pinMode(HUMIDITE_SOL_PIN, INPUT);
    
    // Initialisation pompe
    setupPump();
    Serial.println("✅ Capteurs et pompe initialisés");

    // Initialiser le temps du dernier arrosage
    dernier_arrosage = millis();
    Serial.print("Temps initial : ");
    Serial.println(dernier_arrosage);

    // Initialisation modèle ML
    initMLModel();

    digitalWrite(LED_BUILTIN, LOW);
    Serial.println("🎯 Setup terminé - Démarrage de la boucle principale");
    Serial.flush();
}

// ------------------ LOOP ------------------
void loop() {
    // Clignoter LED rapidement au début de chaque cycle
    digitalWrite(LED_BUILTIN, HIGH);
    
    Serial.println("\n--- Nouvelle inférence ---");
    Serial.flush();

    // Lecture des capteurs
    float T = lireTemperature();
    float Hsol = lireHumiditeSol();
    float PH = lirePH();
    float Lux = lireLuminosite();
    
    // Calcul heures depuis dernier arrosage
    unsigned long temps_ecoule = millis() - dernier_arrosage;
    float Hdep = temps_ecoule / 3600000.0; // conversion ms -> heures

    Serial.printf("T=%.1f°C | Hsol=%.1f%% | PH=%.2f | Lux=%.0f lx | Hdep=%.1fh\n",
                  T, Hsol, PH, Lux, Hdep);
    Serial.flush();

    // ===== Normalisation =====
    float input[5] = {
        (T - 10.0) / 30.0,      // Température normalisée
        Hsol / 100.0,           // Humidité sol normalisée
        Hdep / 72.0,            // Temps écoulé normalisé (max 72h = 3 jours)
        PH / 14.0,              // pH normalisé
        Lux / 2000.0            // Luminosité normalisée
    };

    // Vérification dimension modèle
    if (input_tensor->dims->data[1] != 5) {
        Serial.print("❌ Erreur: Le modèle attend ");
        Serial.print(input_tensor->dims->data[1]);
        Serial.println(" entrées, pas 5 !");
        Serial.flush();
        digitalWrite(LED_BUILTIN, LOW);
        delay(5000);
        return;
    }

    // Copie des données d'entrée
    for (int i = 0; i < 5; i++) {
        input_tensor->data.f[i] = input[i];
    }

    // Inférence
    Serial.println("🔄 Exécution inférence ML...");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.print("❌ Erreur TFLM invoke : ");
        Serial.println(invoke_status);
        Serial.flush();
        digitalWrite(LED_BUILTIN, LOW);
        delay(5000);
        return;
    }

    float volume = output_tensor->data.f[0] * 100.0;
    Serial.printf("🔮 Volume prédit : %.2f cl\n", volume);

    // Calcul durée irrigation
    float debit = 10.0; // cl/seconde
    int duree = (int)(volume / debit);
    duree = constrain(duree, 0, 300); // Max 5 minutes

    Serial.printf("⏱️ Durée irrigation : %d secondes\n", duree);

    // Contrôle irrigation
    if (duree > 2) {
        Serial.println("🚰 Démarrage irrigation...");
        pompeON();
        delay(duree * 1000);
        pompeOFF();

        dernier_arrosage = millis();
        Serial.println("✅ Irrigation terminée");
    } else {
        pompeOFF();
        Serial.println("⏸️ Irrigation non nécessaire");
    }

    Serial.flush();
    Serial.println("😴 Passe 10 minutes...");
    
    // Clignotement LED pendant l'attente
    digitalWrite(LED_BUILTIN, LOW);
    
    // Attente de 10 minutes (600 secondes)
    unsigned long debut_attente = millis();
    while (millis() - debut_attente < 600000) { // 600000 ms = 10 minutes
        // Clignoter toutes les 2 secondes
        if ((millis() / 2000) % 2 == 0) {
            digitalWrite(LED_BUILTIN, HIGH);
        } else {
            digitalWrite(LED_BUILTIN, LOW);
        }
        delay(100);
    }
}