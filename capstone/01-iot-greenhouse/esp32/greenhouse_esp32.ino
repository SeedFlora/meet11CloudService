/*
  Smart Greenhouse - ESP32 asli (pengganti container "device").
  Mengirim JSON yang SAMA dengan device/sim.py ke topik greenhouse/<DEVICE_ID>/telemetry tiap 1 detik.

  STATUS: contoh rujukan. Belum dikompilasi dan belum diuji di perangkat fisik oleh penyusun modul.
          Yang sudah diuji end-to-end adalah ESP32 virtual (device/sim.py) + broker + server + dashboard.

  Perangkat keras (contoh):
    - ESP32 DevKit v1
    - DHT22 (suhu & kelembapan udara)        -> data ke GPIO 4, resistor pull-up 10 kOhm ke 3V3
    - Sensor kelembapan tanah kapasitif v1.2 -> AOUT ke GPIO 34 (ADC1, aman dipakai bersama Wi-Fi)

  Library (Arduino IDE > Library Manager):
    - "PubSubClient" oleh Nick O'Leary
    - "DHT sensor library" oleh Adafruit (+ "Adafruit Unified Sensor")
  Board: "ESP32 Dev Module" (Boards Manager: esp32 oleh Espressif)

  ==== SESUAIKAN DI DEVICE-MU (lihat juga README) =========================================
*/
#include <WiFi.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <time.h>

const char* WIFI_SSID     = "NAMA_WIFI";        // SESUAIKAN: Wi-Fi 2,4 GHz (ESP32 tidak bisa 5 GHz)
const char* WIFI_PASSWORD = "PASSWORD_WIFI";    // SESUAIKAN: jangan di-commit ke Git
const char* MQTT_HOST     = "192.168.1.10";     // SESUAIKAN: IP LAPTOP yang menjalankan docker compose (BUKAN localhost)
const uint16_t MQTT_PORT  = 1883;               // SESUAIKAN bila port broker diubah di docker-compose.yml
const char* DEVICE_ID     = "gh-01";            // satu ID unik per ESP32

const int PIN_DHT   = 4;
const int PIN_TANAH = 34;
const int ADC_KERING = 3200;                    // SESUAIKAN: baca nilai ADC saat sensor di udara (kering)
const int ADC_BASAH  = 1300;                    // SESUAIKAN: baca nilai ADC saat sensor di gelas air (basah)
// =========================================================================================

DHT dht(PIN_DHT, DHT22);
WiFiClient net;
PubSubClient mqtt(net);
char topikTelemetri[64];
float suhu = NAN, lembap = NAN;
unsigned long terakhirDht = 0, terakhirKirim = 0;

void sambungWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf(" OK, IP ESP32: %s\n", WiFi.localIP().toString().c_str());
  configTime(0, 0, "pool.ntp.org");             // ts dikirim dalam detik UTC (epoch), sama dengan sim.py
}

void sambungMqtt() {
  while (!mqtt.connected()) {
    String id = String("esp32-") + DEVICE_ID;
    Serial.printf("MQTT %s:%u ... ", MQTT_HOST, MQTT_PORT);
    if (mqtt.connect(id.c_str())) {
      Serial.println("terhubung");
    } else {
      Serial.printf("gagal (state %d), coba lagi 2 detik\n", mqtt.state());
      delay(2000);                              // cek: IP laptop, firewall port 1883, laptop & ESP32 satu jaringan
    }
  }
}

float persenTanah() {
  int raw = analogRead(PIN_TANAH);
  float p = 100.0f * (ADC_KERING - raw) / (ADC_KERING - ADC_BASAH);
  return constrain(p, 0.0f, 100.0f);
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  analogReadResolution(12);
  snprintf(topikTelemetri, sizeof(topikTelemetri), "greenhouse/%s/telemetry", DEVICE_ID);
  sambungWifi();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) sambungWifi();
  if (!mqtt.connected()) sambungMqtt();
  mqtt.loop();

  unsigned long now = millis();
  if (now - terakhirDht >= 2000) {              // DHT22 hanya boleh dibaca tiap >= 2 detik
    terakhirDht = now;
    float s = dht.readTemperature(), h = dht.readHumidity();
    if (!isnan(s) && !isnan(h)) { suhu = s; lembap = h; }
  }
  if (now - terakhirKirim >= 1000 && !isnan(suhu)) {   // server menghitung fitur dengan asumsi 1 data/detik
    terakhirKirim = now;
    time_t ts = time(nullptr);                  // 0-an bila NTP belum sinkron: server memakai jam server
    char payload[160];
    snprintf(payload, sizeof(payload),
             "{\"device\":\"%s\",\"ts\":%ld,\"suhu\":%.2f,\"kelembapan\":%.2f,\"tanah\":%.2f}",
             DEVICE_ID, (long)ts, suhu, lembap, persenTanah());
    mqtt.publish(topikTelemetri, payload);
    Serial.println(payload);
  }
}
